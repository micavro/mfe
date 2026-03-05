#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多请求 Client：submit/status，从 parquet 数据集读取并测试。"""

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
from mfe.scripts.process_datasets import PROCESSORS

DATASET_NAMES = ("drop", "gsm8k", "hotpotqa", "math")


def load_questions_from_parquet(
    dataset_name: str,
    data_dir: str,
    yaml_name: str = "adv_reason_3.yaml",
    n: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """从 mfe/data/{dataset}/{dataset}.parquet 读取前 n 个问题。"""
    if dataset_name not in PROCESSORS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(PROCESSORS.keys())}")
    rows = PROCESSORS[dataset_name](data_dir, n)
    if not yaml_name.endswith(".yaml"):
        yaml_name = f"{yaml_name}.yaml"
    for r in rows:
        r["yaml"] = yaml_name
    return rows


def _zero_timestamps(results: List[Dict[str, Any]]) -> None:
    """将所有 benchmark、arrive_time、done_time 减去最小值，使时间从 0 开始。"""
    all_ts: List[float] = []
    for r in results:
        at = r.get("arrive_time")
        if at is not None:
            all_ts.append(float(at))
        dt = r.get("done_time")
        if dt is not None:
            all_ts.append(float(dt))
        for vals in (r.get("benchmark") or {}).values():
            if isinstance(vals, (list, tuple)) and len(vals) >= 2:
                all_ts.extend([float(vals[0]), float(vals[1])])
    if not all_ts:
        return
    min_ts = min(all_ts)
    for r in results:
        if r.get("arrive_time") is not None:
            r["arrive_time"] = float(r["arrive_time"]) - min_ts
        if r.get("done_time") is not None:
            r["done_time"] = float(r["done_time"]) - min_ts
        bench = r.get("benchmark") or {}
        r["benchmark"] = {k: [float(v[0]) - min_ts, float(v[1]) - min_ts] for k, v in bench.items()}


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
                mfe_answer = _extract_final_answer(st)
                arrive_time = st.get("arrive_time")
                done_time = st.get("done_time")
                latency = (float(done_time) - float(arrive_time)) if (arrive_time is not None and done_time is not None) else None
                # 题目信息放最前，运行答案命名为 mfe_answer
                out_item: Dict[str, Any] = {
                    "question": item.get("question", ""),
                    "yaml": item.get("yaml", ""),
                    "gold_answer": item.get("gold_answer", ""),
                }
                for k, v in item.items():
                    if k not in out_item:
                        out_item[k] = v
                out_item["mfe_answer"] = mfe_answer
                out_item["benchmark"] = st.get("benchmark", {})
                out_item["total_answer_time"] = st.get("total_answer_time")
                out_item["arrive_time"] = arrive_time
                out_item["done_time"] = done_time
                out_item["latency"] = latency
                out_item["uid"] = uid
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
    p.add_argument("--dataset", required=True, choices=DATASET_NAMES, help="数据集：drop, gsm8k, hotpotqa, math")
    p.add_argument("-n", "--num", type=int, default=None, help="使用前 n 个问题测试，不指定则用全部。保存为 {dataset}_result_{n}.json 或 {dataset}_result.json")
    p.add_argument("--templates-dir", default="templates", help="工作流 YAML 目录")
    p.add_argument("--yaml", default="adv_reason_3.yaml", help="YAML 模板")
    p.add_argument("--send-interval", type=float, default=0.0, help="发送间隔（秒）")
    p.add_argument("--worker-delay", type=float, default=None, help="TestWorker 模拟延迟（秒）")
    p.add_argument("--test-worker", action="store_true", help="使用 TestWorker")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    if args.verbose:
        set_verbose(True)
        os.environ["MFE_VERBOSE"] = "1"
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
        data_dir = os.path.join(root, "data")
        questions = load_questions_from_parquet(args.dataset, data_dir, args.yaml, args.num)
        if not questions:
            print(f"No data for dataset {args.dataset}")
            return
        print(f"Loaded {len(questions)} questions from mfe/data/{args.dataset}/{args.dataset}.parquet")
        results = run_data_test(client, questions, send_interval=args.send_interval)
        _zero_timestamps(results)
        out_dir = os.path.join(root, "data", args.dataset)
        os.makedirs(out_dir, exist_ok=True)
        out_name = f"{args.dataset}_result_{args.num}.json" if args.num is not None else f"{args.dataset}_result.json"
        out_path = os.path.join(out_dir, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(results)} results to {out_path}")
    finally:
        client.close()
        proc.join(timeout=5.0)


if __name__ == "__main__":
    main()
