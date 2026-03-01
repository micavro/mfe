"""
MFE Optimizer（mfe_v）：多工作流、单请求到达即执行

- 不绑定单一 YAML：按请求的 template 按需加载并缓存 DAG。
- 仅 vLLM Worker + schedule_rr：轮询将 DAG 中各 OP 分配到多 GPU。
- execute_one(query)：用 query.template 对应 DAG 执行该请求，返回带 op_output、benchmark 的 Query。
"""

import os
import queue
import time
import torch
import torch.multiprocessing as mp
from logging import getLogger
from typing import Dict, List, Tuple, Any
from mfe.workers import vLLMWorker, TestWorker
from mfe.parser import load_config, build_ops_from_config
from mfe.schedulers import schedule_rr
from mfe.components import Operator, ExecuteInfo, Query
from mfe.util import _visible_physical_gpu_ids

logger = getLogger(__name__)
logger.setLevel("INFO")


def _worker_process(worker_id: int, physical_gpu_id: int, cmd_queue: mp.Queue, result_queue: mp.Queue, use_test_worker: bool = False) -> None:
    """Worker 进程入口：创建 vLLMWorker 或 TestWorker 并运行命令循环。"""
    worker_cls = TestWorker if use_test_worker else vLLMWorker
    worker = worker_cls(worker_id, physical_gpu_id, cmd_queue, result_queue)
    worker.run()


class Optimizer:
    """
    多工作流优化器：每请求可带不同 template（YAML），单请求到达即执行整 DAG。

    __init__(templates_dir)：仅创建 Worker 进程与队列，不加载 DAG。
    _get_dag(template)：解析路径，按需 load_config + build_ops_from_config 并缓存。
    execute_one(query)：取 query.template 的 DAG → schedule_rr → 派发并收集结果，返回执行后的 Query。
    """

    def __init__(self, templates_dir: str = "templates", use_test_worker: bool | None = None, verbose: bool = False, **kwargs) -> None:
        self.templates_dir = os.path.abspath(templates_dir)
        self._template_cache: Dict[str, Tuple[Dict[str, Operator], List[Operator], List[Operator], Any]] = {}

        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        # 是否使用测试用假 Worker（不依赖 vLLM/GPU，输出=输入）
        if use_test_worker is None:
            use_test_worker = os.environ.get("MFE_USE_TEST_WORKER", "").lower() in ("1", "true", "yes")
        self._use_test_worker = use_test_worker
        self._verbose = bool(verbose or os.environ.get("MFE_VERBOSE", "").lower() in ("1", "true", "yes"))

        self.device_cnt = torch.cuda.device_count()
        if self._use_test_worker and self.device_cnt == 0:
            self.device_cnt = 1  # 无 GPU 时至少 1 个 Worker 进程以便跑通流程
        self.processes: List[mp.Process] = []
        self.cmd_queues: List[mp.Queue] = []
        self.result_queues: List[mp.Queue] = []

        for _ in range(self.device_cnt):
            self.cmd_queues.append(mp.Queue())
            self.result_queues.append(mp.Queue())

        phys_ids = _visible_physical_gpu_ids()
        if not phys_ids and not self._use_test_worker:
            raise RuntimeError("No visible GPUs. Set CUDA_VISIBLE_DEVICES or ensure GPUs are available.")
        phys_ids = phys_ids[: self.device_cnt] if phys_ids else list(range(self.device_cnt))

        for i, gpu_id in enumerate(phys_ids):
            proc = mp.Process(
                target=_worker_process,
                args=(i, gpu_id, self.cmd_queues[i], self.result_queues[i], self._use_test_worker),
                daemon=False,
            )
            self.processes.append(proc)
            proc.start()

        # 占位；execute_one 前由 _get_dag 填充
        self.ops: Dict[str, Operator] = {}
        self.start_ops: List[Operator] = []
        self.end_ops: List[Operator] = []
        self.models: Any = set()
        self.queries: List[Query] = []
        self.req_id_map: Dict[Any, Query] = {}
        self.workflows: List[List[Dict]] = []

        logger.info(
            "OptimizerMFE initialized (templates_dir=%s, devices=%d, use_test_worker=%s)",
            self.templates_dir, self.device_cnt, self._use_test_worker,
        )

    def _resolve_template_path(self, template: str) -> str:
        """将 template（文件名或路径）解析为绝对路径，不存在则抛 FileNotFoundError。"""
        template = (template or "").strip()
        if not template:
            raise ValueError("template is empty")
        if os.path.isabs(template) or os.path.sep in template:
            path = os.path.abspath(template)
        else:
            path = os.path.join(self.templates_dir, template)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"template not found: {path}")
        return path

    def _get_dag(self, template: str) -> Tuple[Dict[str, Operator], List[Operator], List[Operator], Any]:
        """按 template 加载并缓存 DAG，返回 (ops, start_ops, end_ops, models)。"""
        path = self._resolve_template_path(template)
        if path in self._template_cache:
            return self._template_cache[path]
        config = load_config(path)
        ops, start_ops, end_ops, models = build_ops_from_config(config)
        self._template_cache[path] = (ops, start_ops, end_ops, models)
        return ops, start_ops, end_ops, models

    def _optimize_queries(self, queries: List[Query]) -> None:
        """按优先级降序、prompt_len 升序排序。"""
        self.queries = sorted(queries, key=lambda x: (-x.priority, x.prompt_len))

    def schedule(self, queries: List[Query], strategy: str = "rr") -> None:
        """仅支持 "rr"：调用 schedule_rr 生成 self.workflows。"""
        self._optimize_queries(queries)
        self.req_id_map = {q.id: q for q in self.queries}
        if strategy != "rr":
            raise ValueError("OptimizerMFE only supports strategy='rr'")
        self.workflows = schedule_rr(self.device_cnt, list(self.ops.values()), self.queries)

    def execute_one(self, query: Query) -> Query:
        """
        用 query.template 对应的 DAG 执行该请求：_get_dag → schedule_rr → 派发任务、收集结果，更新 query.op_output / benchmark，返回该 Query。
        单请求执行期间不向 Worker 发 exit，Worker 常驻以供下一请求使用。
        """
        self.ops, self.start_ops, self.end_ops, self.models = self._get_dag(query.template)
        self.queries = [query]
        self.req_id_map = {query.id: query}
        self.schedule(self.queries, strategy="rr")
        if self._verbose:
            print(f"[OPT] query id={str(query.id)[:8]} template={query.template} DAG_ops={list(self.ops.keys())}")

        finish_flags = [False] * self.device_cnt
        inflight = [False] * self.device_cnt
        worker_pointer = [0] * self.device_cnt

        def _cmd_transfer(task: Dict) -> Dict:
            """将 (op, query_ids) 转为 ExecuteInfo：从 req_id_map 取 prompt + 父 OP 输出拼接。"""
            if task.get("command") != "execute":
                return task
            op, query_ids = task["params"][0], task["params"][1]
            prompts = []
            for qid in query_ids:
                q = self.req_id_map[qid]
                prompt = q.prompt
                if isinstance(prompt, list):
                    prompt = prompt[q.step]
                history_seqs = [self.req_id_map[qid].op_output.get(inp.id, "") for inp in op.input_ops]
                history = "".join(history_seqs)
                prompts.append(prompt + history)
            exe = ExecuteInfo(op=op, query_ids=query_ids, prompts=prompts)
            task = dict(task)
            task["params"] = (exe,)
            return task

        def _task_ready(task: Dict) -> bool:
            """execute 任务：检查所有 query 的父 OP 输出是否已就绪。"""
            if task.get("command") != "execute":
                return True
            op, query_ids = task["params"][0], task["params"][1]
            if not getattr(op, "input_ops", None):
                return True
            parent_ids = [p.id for p in op.input_ops]
            for qid in query_ids:
                q = self.req_id_map[qid]
                for pid in parent_ids:
                    if pid not in q.op_output:
                        return False
            return True

        def _try_send(i: int) -> None:
            """若 Worker i 空闲且下一任务依赖已满足，则派发该任务。"""
            if finish_flags[i] or inflight[i]:
                return
            if worker_pointer[i] >= len(self.workflows[i]):
                finish_flags[i] = True
                return
            task = self.workflows[i][worker_pointer[i]]
            if _task_ready(task):
                if self._verbose and task.get("command") == "execute":
                    op, qids = task["params"][0], task["params"][1]
                    print(f"[OPT] -> Worker {i} op={op.id} query_ids={qids} t={time.perf_counter():.3f}")
                self.cmd_queues[i].put(_cmd_transfer(task))
                inflight[i] = True

        for i in range(self.device_cnt):
            _try_send(i)

        while not all(finish_flags):
            made_progress = False
            for i in range(self.device_cnt):
                if finish_flags[i] or not inflight[i]:
                    continue
                try:
                    message = self.result_queues[i].get(timeout=0.1)
                except queue.Empty:
                    continue
                inflight[i] = False
                made_progress = True
                if not isinstance(message, dict):
                    logger.warning("Worker %d returned non-dict message: %r", i, message)
                    worker_pointer[i] += 1
                    _try_send(i)
                    continue
                cmd = message.get("command")
                if cmd == "execute":
                    result = message.get("result", {})
                    op_name = result.get("op_name") or result.get("node_name")
                    if self._verbose and op_name is not None:
                        qids = [rec["id"] for rec in result.get("item", [])]
                        print(f"[OPT] <- Worker {i} op_name={op_name} query_ids={qids} t={time.perf_counter():.3f}")
                    if op_name is not None:
                        for rec in result.get("item", []):
                            q = self.req_id_map[rec["id"]]
                            q.op_output[op_name] = rec["output"]
                            q.step += 1
                            q.benchmark[op_name] = rec["benchmark"]
                        if "benchmark" in result and op_name in self.ops:
                            self.ops[op_name].benchmark.update(result["benchmark"])
                elif cmd == "error":
                    logger.error("Worker %d error: %s", i, message.get("result"))
                    finish_flags[i] = True  # 避免因依赖未就绪而永远不发送后续任务导致死循环
                worker_pointer[i] += 1
                _try_send(i)
                for j in range(self.device_cnt):
                    if not finish_flags[j]:
                        _try_send(j)
            if not made_progress:
                for j in range(self.device_cnt):
                    if not finish_flags[j]:
                        _try_send(j)

        return self.queries[0]

    def exit(self) -> None:
        """向各 Worker 发 exit，关闭队列并 join 进程。"""
        for i in range(self.device_cnt):
            self.cmd_queues[i].put(("exit", ()))
        for q in self.cmd_queues + self.result_queues:
            q.close()
            q.join_thread()
        for p in self.processes:
            p.join()
        logger.info("OptimizerMFE exited")
