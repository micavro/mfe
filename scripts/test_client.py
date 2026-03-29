#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证：同一批 GSM8K 题目上，每条任务随机指定不同 YAML，各自按模板路径执行。
默认取 GSM8K 前 10 条，开启 verbose 便于终端跟进。
"""

from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import random
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_script_dir)
sys.path.insert(0, os.path.dirname(_root) if os.path.isfile(os.path.join(_root, "__init__.py")) else _root)

from mfe.serve import run_server
from mfe.config import set_verbose
from mfe.scripts.client import (
    Client,
    _json_default,
    _to_json_safe,
    _zero_timestamps,
    load_questions_from_parquet,
    run_data_test,
)


def _list_yaml_templates(templates_dir: str) -> list[str]:
    """templates 目录下所有 .yaml，按文件名排序。"""
    pattern = os.path.join(templates_dir, "*.yaml")
    paths = sorted(glob.glob(pattern))
    return [os.path.basename(p) for p in paths]


def main() -> None:
    p = argparse.ArgumentParser(
        description="GSM8K 前 N 题，每条随机 YAML + verbose，验证多模板路径"
    )
    p.add_argument("-n", "--num", type=int, default=10, help="GSM8K 前 N 题（默认 10）")
    p.add_argument(
        "--templates-dir",
        default="templates",
        help="工作流 YAML 目录（相对 mfe 根目录）",
    )
    p.add_argument("--seed", type=int, default=None, help="随机分配 YAML 的种子，便于复现")
    p.add_argument("--send-interval", type=float, default=0.0, help="题目提交间隔（秒）")
    p.add_argument("--worker-delay", type=float, default=None, help="TestWorker 模拟延迟（秒）")
    p.add_argument("--test-worker", action="store_true", help="使用 TestWorker 而非真实推理")
    p.add_argument("--no-verbose", action="store_true", help="关闭详细日志（默认开启 verbose）")
    args = p.parse_args()

    print("test_client: starting", flush=True)

    if not args.no_verbose:
        set_verbose(True)
        os.environ["MFE_VERBOSE"] = "1"
    # TestWorker 默认每步 sleep 20s（见 workers/worker_test.py），不设则像「卡住无输出」
    if args.test_worker:
        if args.worker_delay is not None:
            os.environ["MFE_TEST_WORKER_DELAY"] = str(args.worker_delay)
        else:
            os.environ.setdefault("MFE_TEST_WORKER_DELAY", "0")
    elif args.worker_delay is not None:
        os.environ["MFE_TEST_WORKER_DELAY"] = str(args.worker_delay)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    templates_abs = os.path.abspath(args.templates_dir)

    yamls = _list_yaml_templates(templates_abs)
    if not yamls:
        print(f"No .yaml found under {templates_abs}")
        sys.exit(1)

    rng = random.Random(args.seed)
    data_dir = os.path.join(root, "data")

    # 占位 yaml，随后逐条覆盖为随机模板
    questions = load_questions_from_parquet("gsm8k", data_dir, yamls[0], args.num)
    if not questions:
        print("No gsm8k data. Ensure mfe/data/gsm8k/gsm8k.parquet exists.")
        sys.exit(1)

    for item in questions:
        item["yaml"] = rng.choice(yamls)

    print(
        f"Loaded {len(questions)} GSM8K questions; "
        f"YAML pool ({len(yamls)}): {', '.join(yamls)}",
        flush=True,
    )
    if not args.no_verbose:
        for i, item in enumerate(questions):
            print(f"  [{i+1}] yaml={item['yaml']}", flush=True)

    req_q = mp.Queue()
    resp_q = mp.Queue()
    proc = mp.Process(
        target=run_server,
        args=(req_q, resp_q, templates_abs, args.test_worker),
        daemon=False,
    )
    proc.start()
    client = Client(req_q, resp_q)
    try:
        results = run_data_test(client, questions, send_interval=args.send_interval)
        _zero_timestamps(results)
        results = _to_json_safe(results)
        out_dir = os.path.join(root, "data", "gsm8k")
        os.makedirs(out_dir, exist_ok=True)
        seed_part = f"_seed{args.seed}" if args.seed is not None else ""
        out_name = f"gsm8k_random_yaml_result_{args.num}{seed_part}.json"
        out_path = os.path.join(out_dir, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=_json_default)
        print(f"Saved {len(results)} results to {out_path}", flush=True)
    finally:
        client.close()
        proc.join(timeout=5.0)


if __name__ == "__main__":
    main()
