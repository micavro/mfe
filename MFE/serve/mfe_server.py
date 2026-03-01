"""
MFE Server：双 Queue 单请求执行服务

主进程创建 request_queue / response_queue 后，在子进程或本进程中调用 run_mfe_server；
循环从 request_queue 取请求 (id, prompt, template)，用 OptimizerMFE.execute_one 执行整 DAG，
将结果与性能统计写入 response_queue。收到 None 或 {"command": "exit"} 时退出并关闭 Optimizer。
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from halo.components import Query
from halo.optimizers.mfe_v import OptimizerMFE

logger = logging.getLogger(__name__)


def _build_response_ok(req_id: str, query: Query) -> Dict[str, Any]:
    """根据执行后的 Query 构造成功响应：op_output、benchmark（各节点 [start,end]）、total_answer_time（秒）。"""
    benchmark_dict = {}
    for op_id, t in query.benchmark.items():
        benchmark_dict[op_id] = [float(t[0]), float(t[1])]
    end_times = [t[1] for t in query.benchmark.values()]
    total_answer_time = (max(end_times) - query.create_time) if end_times else 0.0
    return {
        "id": req_id,
        "ok": True,
        "result": {
            "op_output": dict(query.op_output),
            "benchmark": benchmark_dict,
            "total_answer_time": total_answer_time,
        },
        "error": None,
    }


def _build_response_error(req_id: str, message: str) -> Dict[str, Any]:
    """构造失败响应。"""
    return {"id": req_id, "ok": False, "result": None, "error": str(message)}


def run_mfe_server(
    request_queue,
    response_queue,
    templates_dir: str = "templates",
    use_test_worker: bool | None = None,
    verbose: bool = False,
) -> None:
    """
    在当前进程内运行 MFE 服务循环。

    请求格式: {"id": str, "prompt": str, "template": str}（template 为 YAML 文件名或路径）
    响应格式: {"id": str, "ok": bool, "result": {op_output, benchmark, total_answer_time} | None, "error": str | None}
    收到 None 或 {"command": "exit"} 时退出并调用 opt.exit()。
    use_test_worker: 若为 True，使用 TestWorker（不依赖 vLLM/GPU）；默认从环境变量 MFE_USE_TEST_WORKER 读取。
    verbose: 若为 True，打印每条请求/执行完成的中间信息。
    """
    opt = OptimizerMFE(templates_dir=templates_dir, use_test_worker=use_test_worker, verbose=verbose)
    try:
        while True:
            req = request_queue.get()
            if req is None:
                break
            if isinstance(req, dict) and req.get("command") == "exit":
                break
            if not isinstance(req, dict):
                response_queue.put(_build_response_error("", "invalid request: not a dict"))
                continue
            req_id = req.get("id", "")
            prompt = req.get("prompt", "")
            template = req.get("template", "")
            if not template:
                response_queue.put(_build_response_error(req_id, "missing or empty 'template'"))
                continue
            if verbose:
                print(f"[SERVER] recv id={str(req_id)[:8]} template={template} prompt_len={len(prompt or '')}")
            try:
                query = Query(id=req_id, prompt=prompt, template=template)
                query = opt.execute_one(query)
                if verbose:
                    total = (max(query.benchmark[k][1] for k in query.benchmark) - query.create_time) if query.benchmark else 0
                    print(f"[SERVER] done id={str(req_id)[:8]} total_answer_time={total:.3f}s op_output_keys={list(query.op_output.keys())}")
                response_queue.put(_build_response_ok(req_id, query))
            except FileNotFoundError as e:
                response_queue.put(_build_response_error(req_id, f"template not found: {e}"))
            except Exception as e:
                logger.exception("execute_one failed for req_id=%s", req_id)
                response_queue.put(_build_response_error(req_id, str(e)))
    finally:
        opt.exit()
