"""双 Queue 单请求服务：取 (id, prompt, template)，execute_one 整 DAG，写 response。"""

from __future__ import annotations

import logging
from typing import Any, Dict

from halo.components import Query
from halo.optimizers.mfe_v import OptimizerMFE
from mfe.config import is_verbose

logger = logging.getLogger(__name__)


def _build_response_ok(req_id: str, query: Query) -> Dict[str, Any]:
    benchmark_dict = {op_id: [float(t[0]), float(t[1])] for op_id, t in query.benchmark.items()}
    end_times = [t[1] for t in query.benchmark.values()]
    total_answer_time = (max(end_times) - query.create_time) if end_times else 0.0
    return {
        "id": req_id, "ok": True,
        "result": {"op_output": dict(query.op_output), "benchmark": benchmark_dict, "total_answer_time": total_answer_time},
        "error": None,
    }


def _build_response_error(req_id: str, message: str) -> Dict[str, Any]:
    return {"id": req_id, "ok": False, "result": None, "error": str(message)}


def run_mfe_server(
    request_queue, response_queue,
    templates_dir: str = "templates",
    use_test_worker: bool | None = None,
) -> None:
    """循环取请求执行 DAG，写响应。请求 {"id","prompt","template"}；收到 None/exit 退出。"""
    opt = OptimizerMFE(templates_dir=templates_dir, use_test_worker=use_test_worker)
    try:
        while True:
            req = request_queue.get()
            if req is None or (isinstance(req, dict) and req.get("command") == "exit"):
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
            if is_verbose():
                print(f"[SERVER] recv id={str(req_id)[:8]} template={template} prompt_len={len(prompt or '')}")
            try:
                query = Query(id=req_id, prompt=prompt, template=template)
                query = opt.execute_one(query)
                if is_verbose():
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
