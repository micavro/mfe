#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多请求 Client：submit/status/send_test，通过 Queue 与 Server 通信。支持 JSON 输入/输出做数据测试。"""

from __future__ import annotations

import json
import os
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


def send_test(client: Client, templates_dir: str, num_requests: int = 5, send_interval: float = 0.5, templates: Optional[List[str]] = None) -> List[str]:
    if templates is None or len(templates) == 0:
        templates = ["adv_reason_3.yaml"]
    tpls = [t if t.endswith(".yaml") else f"{t}.yaml" for t in templates]
    uids: List[str] = []
    for i in range(num_requests):
        if i > 0:
            time.sleep(send_interval)
        tpl = tpls[i % len(tpls)]
        prompt = f"Question {i+1}"
        uid = client.submit(tpl, prompt)
        uids.append(uid)
    return uids


def _strip_chat_template(raw: str) -> str:
    """从 Llama chat template 原始输出中提取最后一个 assistant 回复。"""
    if not raw:
        return ""
    marker = "assistant\n\n"
    idx = raw.rfind(marker)
    if idx < 0:
        return raw.strip()
    result = raw[idx + len(marker):]
    if result.endswith("<|eot_id|>"):
        result = result[:-10]
    return result.strip()


def _extract_final_answer(st: Dict[str, Any]) -> str:
    """从 status 的 op_output 中取最后节点新增的输出，并剔除 chat template，得到纯文本答案。"""
    op_output = st.get("op_output", {}) or {}
    benchmark = st.get("benchmark", {}) or {}
    if not op_output:
        return ""
    if not benchmark:
        return _strip_chat_template(next(iter(op_output.values()), ""))
    sorted_ops = sorted(benchmark.keys(), key=lambda k: benchmark[k][1] if k in benchmark else 0)
    last_op = sorted_ops[-1]
    full_last = op_output.get(last_op, "")
    if len(sorted_ops) > 1:
        prev_op = sorted_ops[-2]
        prefix = op_output.get(prev_op, "")
        if full_last.startswith(prefix):
            full_last = full_last[len(prefix):].strip()
    return _strip_chat_template(full_last)


def run_data_test(
    client: Client,
    questions: List[Dict[str, Any]],
    send_interval: float = 0.0,
) -> List[Dict[str, Any]]:
    """发送所有问题、等待完成、返回结果列表。questions 每项需有 question、yaml。"""
    uids: List[str] = []
    for i, item in enumerate(questions):
        if i > 0 and send_interval > 0:
            time.sleep(send_interval)
        q = item.get("question", "")
        yaml_name = item.get("yaml", "adv_reason_3.yaml")
        if not yaml_name.endswith(".yaml"):
            yaml_name = f"{yaml_name}.yaml"
        uid = client.submit(yaml_name, q)
        uids.append(uid)

    completed: Dict[str, Dict[str, Any]] = {}
    last_progress = 0.0
    while len(completed) < len(uids):
        for i, uid in enumerate(uids):
            if uid in completed:
                continue
            st = client.status(uid)
            if st and st.get("status") == "completed":
                item = questions[i]
                answer = _extract_final_answer(st)
                out_item: Dict[str, Any] = {
                    "question": item.get("question", ""),
                    "yaml": item.get("yaml", ""),
                    "answer": answer,
                    "op_output": st.get("op_output", {}),
                    "benchmark": st.get("benchmark", {}),
                    "total_answer_time": st.get("total_answer_time"),
                    "uid": uid,
                }
                if "gold_answer" in item:
                    out_item["gold_answer"] = item["gold_answer"]
                completed[uid] = out_item
                if is_verbose() and st.get("total_answer_time") is not None:
                    print(f"  [{i+1}/{len(uids)}] uid={uid[:8]}... completed in {st['total_answer_time']:.2f}s", flush=True)
        now = time.perf_counter()
        if len(completed) < len(uids) and now - last_progress >= 10.0:
            print(f"  ... waiting: {len(completed)}/{len(uids)} completed", flush=True)
            last_progress = now
        time.sleep(0.5)

    return [completed[uid] for uid in uids]


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="MFE 多请求 Client")
    p.add_argument("--templates-dir", default="templates", help="工作流 YAML 目录")
    p.add_argument("--input-json", default=None, help="输入 JSON 文件，每项含 question、yaml")
    p.add_argument("--output-json", default=None, help="输出 JSON 文件，保存答案等")
    p.add_argument("--templates", nargs="+", default=None, help="轮换使用的 YAML 列表（默认测试用）")
    p.add_argument("-n", "--num", type=int, default=5, help="请求数（默认测试用）")
    p.add_argument("--send-interval", type=float, default=0.5, help="发送间隔（秒）")
    p.add_argument("--worker-delay", type=float, default=None, help="TestWorker 模拟延迟（秒），如 2")
    p.add_argument("--test-worker", action="store_true", help="使用 TestWorker")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    if args.verbose:
        set_verbose(True)
        os.environ["MFE_VERBOSE"] = "1"  # 子进程 server/worker 继承
    if args.worker_delay is not None:
        os.environ["MFE_TEST_WORKER_DELAY"] = str(args.worker_delay)

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
        if args.input_json and args.output_json:
            with open(args.input_json, "r", encoding="utf-8") as f:
                questions = json.load(f)
            if not isinstance(questions, list):
                questions = [questions]
            print(f"Loaded {len(questions)} questions from {args.input_json}")
            results = run_data_test(client, questions, send_interval=args.send_interval)
            out_path = os.path.abspath(args.output_json)
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(results)} results to {out_path}")
        else:
            templates_list = args.templates or ["adv_reason_3.yaml", "adv_reason_4m.yaml", "multi_step_retrival.yaml"]
            uids = send_test(
                client, templates_abs,
                num_requests=args.num,
                send_interval=args.send_interval,
                templates=templates_list,
            )
            completed = set()
            while len(completed) < len(uids):
                for uid in uids:
                    if uid in completed:
                        continue
                    st = client.status(uid)
                    if st and st.get("status") == "completed":
                        completed.add(uid)
                        t = st.get("total_answer_time")
                        if t is not None:
                            print(f"  uid={uid[:8]}... completed in {t:.2f}s")
                time.sleep(0.5)
            print(f"Done. {len(completed)}/{len(uids)} completed.")
    finally:
        client.close()
        proc.join(timeout=5.0)


if __name__ == "__main__":
    main()
