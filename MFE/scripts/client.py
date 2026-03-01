#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多请求 Client：submit、status、send_test
"""

from __future__ import annotations

import os
import random
import sys
import threading
import time
import uuid
import multiprocessing as mp
from typing import Any, Dict, List, Optional

_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_script_dir)
sys.path.insert(0, os.path.dirname(_root) if os.path.isfile(os.path.join(_root, "__init__.py")) else _root)

from mfe.serve import run_server
from mfe.config import is_verbose, set_verbose


class Client:
    """通过 Queue 与 Server 进程通信的客户端。"""

    def __init__(
        self,
        request_queue: mp.Queue,
        response_queue: mp.Queue,
    ) -> None:
        self._req_q = request_queue
        self._resp_q = response_queue
        self._pending: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        self._consumer = threading.Thread(target=self._consume, daemon=True)
        self._consumer.start()

    def _consume(self) -> None:
        while True:
            try:
                msg = self._resp_q.get()
            except Exception:
                break
            if not isinstance(msg, dict):
                continue
            req_id = msg.get("req_id", "")
            with self._lock:
                if req_id in self._pending:
                    self._pending[req_id]["result"] = msg
                    self._pending[req_id]["event"].set()

    def submit(self, dag: str, input_text: str) -> str:
        """提交请求，返回 uid。"""
        req_id = str(uuid.uuid4())
        if is_verbose():
            prompt_preview = (input_text or "")[:50] + ("..." if len(input_text or "") > 50 else "")
            print(f"[REQ] id={req_id[:8]} template={dag} prompt={prompt_preview!r}", flush=True)
        ev = threading.Event()
        with self._lock:
            self._pending[req_id] = {"event": ev, "result": None}
        self._req_q.put({"req_id": req_id, "command": "submit", "dag": dag, "input": input_text})
        ev.wait(timeout=10.0)
        with self._lock:
            r = self._pending.pop(req_id, {}).get("result")
        if r and r.get("error") is None:
            return r.get("result", {}).get("uid", "")
        raise RuntimeError(r.get("error", "submit failed") if r else "timeout")

    def status(self, uid: str) -> Optional[Dict[str, Any]]:
        """查询 uid 状态。"""
        req_id = str(uuid.uuid4())
        ev = threading.Event()
        with self._lock:
            self._pending[req_id] = {"event": ev, "result": None}
        self._req_q.put({"req_id": req_id, "command": "status", "uid": uid})
        ev.wait(timeout=5.0)
        with self._lock:
            r = self._pending.pop(req_id, {}).get("result")
        if r and r.get("error") is None:
            return r.get("result")
        return None

    def close(self) -> None:
        self._req_q.put({"command": "exit"})


def send_test(
    client: Client,
    templates_dir: str,
    num_requests: int = 5,
    delay_range: tuple = (1.0, 5.0),
    template: str = "adv_reason_3.yaml",
) -> List[str]:
    """
    模拟随机时间点发送若干请求。
    返回 uid 列表。
    """
    tpl = template if template.endswith(".yaml") else f"{template}.yaml"
    uids: List[str] = []
    for i in range(num_requests):
        time.sleep(random.uniform(*delay_range))
        prompt = f"Question {i+1}"
        uid = client.submit(tpl, prompt)
        uids.append(uid)
    return uids


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="MFE 多请求 Client")
    p.add_argument("--templates-dir", default="templates", help="工作流 YAML 目录")
    p.add_argument("--template", default="adv_reason_3.yaml", help="默认工作流")
    p.add_argument("-n", "--num", type=int, default=5, help="send_test 请求数")
    p.add_argument("--test-worker", action="store_true", help="使用 TestWorker")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    if args.verbose:
        set_verbose(True)

    req_q = mp.Queue()
    resp_q = mp.Queue()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    templates_abs = os.path.abspath(args.templates_dir)

    proc = mp.Process(
        target=run_server,
        args=(req_q, resp_q, templates_abs, args.test_worker),
        daemon=False,
    )
    proc.start()

    client = Client(req_q, resp_q)
    try:
        uids = send_test(client, templates_abs, num_requests=args.num, template=args.template)
        print(f"\nPolling status for {len(uids)} requests...")
        completed = set()
        last_status: Dict[str, str] = {}
        while len(completed) < len(uids):
            for uid in uids:
                if uid in completed:
                    continue
                st = client.status(uid)
                if st:
                    s = st.get("status", "")
                    if last_status.get(uid) != s:
                        last_status[uid] = s
                        print(f"  uid={uid[:8]}... status={s}")
                    if s == "completed":
                        completed.add(uid)
                        t = st.get("total_answer_time")
                        if t is not None:
                            print(f"    total_answer_time={t:.3f}s")
            time.sleep(0.5)
    finally:
        client.close()
        proc.join(timeout=5.0)


if __name__ == "__main__":
    main()
