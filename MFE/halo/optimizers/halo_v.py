"""
Optimizer 模块：Halo 系统的核心协调器

该模块实现了 Optimizer 类，负责：
- 从 YAML 配置构建工作流图
- 创建和管理多个 Worker 进程（每个 GPU 一个）
- 调用调度器生成执行计划
- 协调 Worker 执行任务，保证依赖关系
- 收集结果并更新 Query 状态
- 性能监控和统计

这是整个系统的核心协调层，连接配置解析、调度和执行。
"""

import time
import queue
import numpy as np
import torch
import torch.multiprocessing as mp
from logging import getLogger
from typing import Dict, List

from halo.workers import vLLMWorker
from halo.parser import load_config, build_ops_from_config
from halo.schedulers import schedule_rr
from halo.components import Operator, ExecuteInfo, Query  # types only
from halo.util import _visible_physical_gpu_ids

logger = getLogger(__name__)
logger.setLevel("INFO")


class Optimizer:
    """
    多进程编排器：管理整个工作流的执行生命周期
    
    Optimizer 是 Halo 系统的核心协调器，负责：
    1. 从 YAML 配置构建 Operator 图（DAG）
    2. 为每个可见的物理 GPU 创建一个独立的 Worker 进程
    3. 调用调度器生成执行计划（workflows）
    4. 协调 Worker 执行任务，保证依赖关系
    5. 收集 Worker 的执行结果并更新 Query 状态
    6. 提供性能监控和统计功能
    
    Attributes:
        config: 解析后的配置字典
        device_cnt: 可用 GPU 数量
        processes: Worker 进程列表
        cmd_queues: 每个 Worker 的命令队列列表
        result_queues: 每个 Worker 的结果队列列表
        ops: Operator 字典（op_id -> Operator）
        start_ops: 起始节点列表
        end_ops: 终止节点列表
        models: 所有使用的模型名称集合
        workflows: 执行计划（每个设备上的任务列表）
        queries: 查询列表
        req_id_map: 查询 ID 到 Query 对象的映射
        dp_threshold: 数据并行的阈值（查询数量）
    """

    def __init__(self, config_path: str, **kwargs):
        """
        初始化优化器，加载配置并创建 Worker 进程
        
        Args:
            config_path: YAML 配置文件的路径
            **kwargs: 其他可选参数
        """
        # 加载 YAML 配置
        self.config = load_config(config_path)
        # 设置多进程启动方式为 "spawn"（Windows 兼容，确保每个进程有独立的 Python 解释器）
        mp.set_start_method("spawn", force=True)

        # 检测可用 GPU 数量
        self.device_cnt = torch.cuda.device_count()
        # 初始化进程和队列列表
        self.processes: List[mp.Process] = []      # Worker 进程列表
        self.cmd_queues: List[mp.Queue] = []       # 每个 Worker 的命令队列
        self.result_queues: List[mp.Queue] = []    # 每个 Worker 的结果队列
        self.dp_threshold = 2                      # 数据并行阈值：查询数量超过此值才考虑数据并行

        # 构建 Operator 图（DAG）
        self.ops, self.start_ops, self.end_ops, self.models = build_ops_from_config(self.config)
        # 创建 Worker 进程
        self._create_workers()
        logger.info("Optimizer initialized")

    # ---------------- Workers ---------------- #

    def _create_workers(self) -> None:
        """
        Spawn one worker per visible physical GPU.
        Each worker process is restricted to exactly one physical GPU by
        narrowing CUDA_VISIBLE_DEVICES inside the child process.
        """
        # Per-worker queues
        for _ in range(self.device_cnt):
            self.cmd_queues.append(mp.Queue())
            self.result_queues.append(mp.Queue())

        # Resolve visible physical GPU IDs
        phys_ids = _visible_physical_gpu_ids()
        if not phys_ids:
            raise RuntimeError("No visible GPUs. Set CUDA_VISIBLE_DEVICES or ensure GPUs are available.")
        phys_ids = phys_ids[: self.device_cnt]

        # Spawn processes
        for i, physical_gpu_id in enumerate(phys_ids):
            proc = mp.Process(
                target=worker_process,
                args=(i, physical_gpu_id, self.cmd_queues[i], self.result_queues[i]),
                daemon=False,
            )
            self.processes.append(proc)
            proc.start()

    # ---------------- Query ordering ---------------- #

    def _optimize_queries(self, queries: List[Query]) -> None:
        """
        对查询进行排序优化
        
        排序规则：
        1. 优先级降序（高优先级先执行）
        2. 提示词长度升序（短查询先执行，可能更快完成）
        
        Args:
            queries: 查询列表（会被原地排序）
        """
        self.queries = sorted(queries, key=lambda x: (-x.priority, x.prompt_len))

    # ---------------- Scheduling (choose strategy) ---------------- #

    def schedule(self, queries: List[Query], strategy: str = "rr") -> None:
        """
        根据调度策略生成执行计划。当前仅支持 "rr"（轮询）：按拓扑序将每个 OP 轮询分配到设备。

        workflows：每设备一个任务列表，每项为 {"command": "execute", "params": (op, query_ids)}。
        """
        self._optimize_queries(queries)
        self.req_id_map = {q.id: q for q in self.queries}
        if strategy != "rr":
            raise ValueError("Only strategy='rr' is supported.")
        self.workflows = schedule_rr(self.device_cnt, list(self.ops.values()), self.queries)

    # ---------------- Execution loop ---------------- #

    def execute(self, queries: List[Query] = None, return_queries: bool = False, skip_exit: bool = False):
        """
        向 Worker 派发任务并收集结果
        
        依赖关系保护机制：
        - 在向 Worker 发送任务前，验证该任务中所有查询的所有父 OP 输出都已就绪
        - 如果未就绪，暂不派发；其他任务完成后会重试派发
        
        Args:
            queries: 查询列表，如果为 None 则使用已调度的查询
            return_queries: 如果为 True，返回查询列表；否则返回执行时间
            skip_exit: 如果为 True，不发送退出命令给 Worker
            
        Returns:
            List[Query] 或 float: 查询列表或执行时间（秒）
        """
        if queries is not None:
            self.schedule(queries, strategy="rr")

        # 执行状态跟踪
        finish_flags = [False] * self.device_cnt      # Worker 是否已完成所有任务
        inflight = [False] * self.device_cnt          # Worker 是否有正在执行的任务
        worker_pointer = [0] * self.device_cnt        # 每个 Worker 的下一个任务索引

        def _cmd_transfer(task: Dict) -> Dict:
            """
            将调度器生成的 task 转换为 Worker 可执行的格式
            
            关键逻辑：依赖关系处理
            - 从 Query.op_output 中提取所有父 OP 的输出
            - 将这些输出拼接到当前 prompt 后面
            - 实现多步推理：后续 OP 可以看到前序 OP 的结果
            
            Args:
                task: 调度器生成的任务字典，格式为 {"command": "execute", "params": (op, query_ids)}
                
            Returns:
                Dict: 转换后的任务字典，params 为 (ExecuteInfo,)
            """
            if task["command"] == "execute":
                op, query_ids = task["params"][0], task["params"][1]
                prompts = []
                for qid in query_ids:
                    # 获取查询的原始 prompt
                    prompt = self.req_id_map[qid].prompt
                    # 如果 prompt 是列表（多轮对话），根据 step 选择对应的 prompt
                    if isinstance(prompt, list):
                        step = self.req_id_map[qid].step
                        prompt = prompt[step]
                    
                    # 关键：拼接父节点的输出作为历史
                    # 从 Query.op_output 字典中提取所有父 OP 的输出
                    history_seqs = [
                        self.req_id_map[qid].op_output.get(inp.id, "")
                        for inp in op.input_ops
                    ]
                    history = "".join(history_seqs)  # 将所有父节点的输出拼接
                    prompts.append(prompt + history)  # 将历史拼接到当前 prompt 后面
                
                # 创建 ExecuteInfo 对象，封装执行任务信息
                exe = ExecuteInfo(op=op, query_ids=query_ids, prompts=prompts)
                task["params"] = (exe,)
            return task

        def _task_ready(task: Dict) -> bool:
            """
            检查任务是否满足依赖关系，是否可以派发
            
            这是依赖关系保证机制的核心：在派发任务前，检查该任务的所有父 OP 是否已完成。
            只有所有父 OP 的输出都已就绪，任务才会被派发。
            
            Args:
                task: 任务字典
                
            Returns:
                bool: True 表示任务可以派发，False 表示依赖未满足，需要等待
            """
            # 非 execute 命令总是就绪
            if task.get("command") != "execute":
                return True
            
            op, query_ids = task["params"][0], task["params"][1]
            # 起始节点（无输入）总是就绪
            if not getattr(op, "input_ops", None):
                return True
            
            # 检查所有父 OP 的输出是否都已就绪
            parent_ids = [p.id for p in op.input_ops]
            for qid in query_ids:
                q = self.req_id_map[qid]
                for pid in parent_ids:
                    # 如果某个父 OP 的输出不在 op_output 中，说明依赖未满足
                    if pid not in q.op_output:
                        return False
            return True

        def _try_send(i: int) -> None:
            """
            尝试向 Worker i 发送下一个任务
            
            只有在以下条件都满足时才会发送：
            1. Worker 未完成所有任务
            2. Worker 当前没有正在执行的任务
            3. 还有待执行的任务
            4. 任务的所有依赖都已满足
            
            Args:
                i: Worker 索引（设备索引）
            """
            # 如果 Worker 已完成或正在执行任务，直接返回
            if finish_flags[i] or inflight[i]:
                return

            # 如果该 Worker 的所有任务已完成
            if worker_pointer[i] >= len(self.workflows[i]):
                finish_flags[i] = True  # 标记为完成
                if not skip_exit:
                    self.cmd_queues[i].put(("exit", ()))  # 发送退出命令
                return

            # 获取当前任务
            task = self.workflows[i][worker_pointer[i]]
            # 检查依赖是否满足
            if _task_ready(task):
                # 转换任务格式并发送到 Worker
                self.cmd_queues[i].put(_cmd_transfer(task))
                inflight[i] = True  # 标记为正在执行

        exe_start = time.perf_counter()

        # 初始派发：尝试向所有 Worker 发送第一个任务
        for i in range(self.device_cnt):
            _try_send(i)

        # 收集循环：持续收集结果并派发新任务
        while not all(finish_flags):
            made_progress = False
            for i in range(self.device_cnt):
                # 如果 Worker 已完成或空闲（没有正在执行的任务），跳过
                if finish_flags[i] or not inflight[i]:
                    continue

                try:
                    message = self.result_queues[i].get(timeout=0.1)
                except queue.Empty:
                    continue

                # 收到消息 -> Worker i 的正在执行任务已完成
                inflight[i] = False
                made_progress = True

                # 防御性检查：确保消息是字典格式
                if not isinstance(message, dict):
                    logger.warning("Worker %d returned non-dict message: %r", i, message)
                    # 推进指针以避免因格式错误的消息导致死锁
                    worker_pointer[i] += 1
                    _try_send(i)
                    continue

                cmd = message.get("command")
                if cmd == "execute":
                    result = message.get("result", {})
                    op_name = result.get("op_name") or result.get("node_name")  # 兼容旧版本
                    if op_name is None:
                        logger.error("Worker %d result missing op_name/node_name: %r", i, result)
                    else:
                        # 更新每个查询的输出：将 OP 的执行结果写入 Query.op_output
                        # 这是实现工作流依赖关系的关键：后续 OP 可以从 op_output 中获取父 OP 的输出
                        for rec in result.get("item", []):
                            q = self.req_id_map[rec["id"]]
                            q.op_output[op_name] = rec["output"]  # 存储 OP 的输出
                            q.step += 1                           # 执行步骤递增
                            q.benchmark[op_name] = rec["benchmark"]  # 记录性能基准
                        
                        # 更新 OP 的性能指标（累加统计）
                        if "benchmark" in result and op_name in self.ops:
                            self.ops[op_name].benchmark.update(result["benchmark"])

                elif cmd == "error":
                    # 处理错误消息
                    logger.error("Worker %d error: %s", i, message.get("result"))

                # 推进指针并尝试发送下一个任务给该 Worker
                worker_pointer[i] += 1
                _try_send(i)

                # 任何任务完成后，其他 Worker 可能被解锁（依赖满足）
                # 尝试向所有空闲 Worker 派发任务，避免全局阻塞
                for j in range(self.device_cnt):
                    if not finish_flags[j]:
                        _try_send(j)

            # 如果没有进展，再次尝试派发（避免死锁）
            # 这确保了即使某些 Worker 暂时无法派发任务，系统也会继续尝试
            if not made_progress:
                for j in range(self.device_cnt):
                    if not finish_flags[j]:
                        _try_send(j)

        if return_queries:
            return self.queries
        return time.perf_counter() - exe_start

    # ---------------- Metrics & Exit ---------------- #

    def print_latency_percentiles(self):
        """
        计算并打印所有查询的延迟百分位数
        
        计算每个查询从创建到最后一个 OP 完成的总时间，然后统计延迟分布。
        输出 P50（中位数）和 P95（95 百分位），用于性能分析。
        
        Note:
            - P50（中位数）：50% 的查询延迟低于此值
            - P95（95 百分位）：95% 的查询延迟低于此值
        """
        all_latencies = []
        for q in self.req_id_map.values():
            start_time = q.create_time  # 查询创建时间
            points = list(q.benchmark.values())  # 所有 OP 的执行时间范围
            if not points:
                continue
            # 最后一个 benchmark 的结束时间就是查询完成时间
            end_time = points[-1][-1]
            # 计算总延迟
            all_latencies.append(end_time - start_time)
        
        if not all_latencies:
            return
        
        # 计算百分位数
        p50 = float(np.percentile(all_latencies, 50))  # 中位数
        p95 = float(np.percentile(all_latencies, 95))  # 95 百分位
        print(f"Latency Percentiles: P50={p50:.3f}s, P95={p95:.3f}s")

    def exit(self):
        """
        关闭所有队列并等待进程结束
        
        清理资源：
        1. 关闭所有进程间通信队列
        2. 等待所有队列线程结束
        3. 等待所有 Worker 进程结束
        """
        # 关闭所有队列
        for q in self.cmd_queues + self.result_queues:
            q.close()        # 关闭队列
            q.join_thread()  # 等待队列线程结束
        # 等待所有 Worker 进程结束
        for p in self.processes:
            p.join()
        logger.info("Optimizer exited")


def worker_process(id, device, cmd_queue, result_queue):
    """
    Worker 进程的入口点
    
    这是每个 Worker 进程的入口函数，由 Optimizer 在创建进程时调用。
    函数会创建 vLLMWorker 实例并启动命令循环，持续监听命令队列并执行任务。
    
    Args:
        id: Worker ID（逻辑 ID，用于标识）
        device: 物理 GPU ID（Worker 会绑定到此 GPU）
        cmd_queue: 命令队列，Optimizer 通过此队列发送任务
        result_queue: 结果队列，Worker 通过此队列返回结果
    """
    worker = vLLMWorker(id, device, cmd_queue, result_queue)
    worker.run()  # 启动命令循环


if __name__ == "__main__":
    import os, logging

    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        filename="logs/halo.log",
        filemode="w",
        format="%(asctime)s %(processName)s[%(process)d] %(levelname)s %(name)s: %(message)s",
    )

    from halo.components import Query

    opt = Optimizer("templates/adv_reason_3.yaml")
    queries = [Query(i, "What is Machine Learning System?") for i in range(8)]
    opt.schedule(queries)
    queries = opt.execute(return_queries=True)
    for q in queries:
        print(f"Query {q.id} result: {q.op_output}")
        break

    for op in opt.ops.values():
        print(f"Op {op.id} benchmark: {op.benchmark}")
    opt.exit()
