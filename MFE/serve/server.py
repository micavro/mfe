"""
多请求异步 Server：submit/status API，共享 GPU 池
"""

from __future__ import annotations

import multiprocessing as mp
import time
from queue import Empty
from typing import Any, Dict, Optional

from mfe.optimizers.multi_request import MultiRequestOptimizer
from mfe.config import is_verbose


def run_server(
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    templates_dir: str = "templates",
    use_test_worker: bool | None = None,
) -> None:
    """
    在子进程中运行多请求 Server。
    请求格式: {"req_id": str, "command": "submit"|"status"|"exit", ...}
    响应格式: {"req_id": str, "result": ...}
    """
    opt = MultiRequestOptimizer(
        templates_dir=templates_dir,
        use_test_worker=use_test_worker,
    )
    try:
        while True:
            try:
                req = request_queue.get(timeout=0.5)
            except Empty:
                continue
            if req is None:
                break
            if isinstance(req, dict) and req.get("command") == "exit":
                break
            if not isinstance(req, dict):
                response_queue.put({"req_id": "", "result": None, "error": "invalid request"})
                continue

            req_id = req.get("req_id", "")
            cmd = req.get("command", "")

            if cmd == "submit":
                dag = req.get("dag", "")
                input_text = req.get("input", "")
                if is_verbose():
                    print(f"[SERVER] recv id={req_id[:8]} template={dag} prompt_len={len(input_text or '')}", flush=True)
                try:
                    uid = opt.submit(dag, input_text)
                    if is_verbose():
                        print(f"[SERVER] submit ok uid={uid[:8]}...", flush=True)
                    response_queue.put({"req_id": req_id, "result": {"uid": uid}, "error": None})
                except Exception as e:
                    response_queue.put({"req_id": req_id, "result": None, "error": str(e)})
            elif cmd == "status":
                uid = req.get("uid", "")
                st = opt.status(uid)
                response_queue.put({"req_id": req_id, "result": st, "error": None})
            else:
                response_queue.put({"req_id": req_id, "result": None, "error": f"unknown command: {cmd}"})
    finally:
        opt.exit()
