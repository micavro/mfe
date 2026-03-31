#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""调度基线对比：parallel（当前调度） vs serial（严格串行）。"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from statistics import mean
from typing import Any, Dict, List, Optional

from mfe.config import set_verbose
from mfe.serve import run_server
from mfe.scripts.client import (
    Client,
    _extract_final_answer,
    _json_default,
    _to_json_safe,
    _zero_timestamps,
    load_questions_from_parquet,
    run_data_test,
)


def _percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    arr = sorted(values)
    if len(arr) == 1:
        return float(arr[0])
    pos = (len(arr) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(arr) - 1)
    frac = pos - lo
    return float(arr[lo] * (1 - frac) + arr[hi] * frac)


def _calc_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [r for r in results if r.get("done_time") is not None and r.get("arrive_time") is not None]
    latencies = [float(r["latency"]) for r in valid if r.get("latency") is not None]
    services = [float(r["service_time"]) for r in valid if r.get("service_time") is not None]
    idles = [float(r["idle_time"]) for r in valid if r.get("idle_time") is not None]
    runs = [float(r["run_time"]) for r in valid if r.get("run_time") is not None]

    if valid:
        min_arrive = min(float(r["arrive_time"]) for r in valid)
        max_done = max(float(r["done_time"]) for r in valid)
        makespan = max_done - min_arrive
    else:
        makespan = 0.0

    count = len(results)
    completed = len(valid)
    throughput = (completed / makespan) if makespan > 0 else 0.0
    return {
        "count": count,
        "completed": completed,
        "success_rate": (completed / count) if count > 0 else 0.0,
        "makespan": makespan,
        "throughput": throughput,
        "latency_mean": (mean(latencies) if latencies else None),
        "latency_p50": _percentile(latencies, 0.50),
        "latency_p95": _percentile(latencies, 0.95),
        "latency_p99": _percentile(latencies, 0.99),
        "service_time_mean": (mean(services) if services else None),
        "service_time_p50": _percentile(services, 0.50),
        "service_time_p95": _percentile(services, 0.95),
        "idle_time_mean": (mean(idles) if idles else None),
        "idle_time_p50": _percentile(idles, 0.50),
        "idle_time_p95": _percentile(idles, 0.95),
        "run_time_mean": (mean(runs) if runs else None),
        "run_time_p50": _percentile(runs, 0.50),
        "run_time_p95": _percentile(runs, 0.95),
    }


def _build_result_item(item: Dict[str, Any], uid: str, st: Dict[str, Any], submit_time: float) -> Dict[str, Any]:
    bench = st.get("benchmark") or {}
    start_time = min(float(v[0]) for v in bench.values()) if bench else None
    arrive_time = st.get("arrive_time")
    done_time = st.get("done_time")
    idle_time = (float(start_time) - float(arrive_time)) if (start_time is not None and arrive_time is not None) else None
    latency = (float(done_time) - float(arrive_time)) if (done_time is not None and arrive_time is not None) else None
    service_time = (float(done_time) - float(start_time)) if (done_time is not None and start_time is not None) else None
    op_durations = {
        op_name: (float(v[1]) - float(v[0]))
        for op_name, v in bench.items()
        if isinstance(v, (list, tuple)) and len(v) >= 2
    }
    end_op_name = max(bench.keys(), key=lambda k: bench[k][1]) if bench else None
    q_full = item.get("question", "")

    out_item: Dict[str, Any] = {
        "question_preview": q_full[:100] if isinstance(q_full, str) else str(q_full)[:100],
        "question": q_full,
        "yaml": item.get("yaml", ""),
    }
    for k, v in item.items():
        if k not in out_item:
            out_item[k] = v

    out_item["mfe_answer"] = _extract_final_answer(st)
    out_item["benchmark"] = bench
    out_item["op_durations"] = op_durations
    out_item["run_time"] = sum(op_durations.values())
    out_item["end_op_name"] = end_op_name
    out_item["worker_assignments"] = st.get("worker_assignments") or {}
    out_item["submit_time"] = submit_time
    out_item["total_answer_time"] = st.get("total_answer_time")
    out_item["arrive_time"] = arrive_time
    out_item["start_time"] = start_time
    out_item["service_time"] = service_time
    out_item["idle_time"] = idle_time
    out_item["done_time"] = done_time
    out_item["latency"] = latency
    out_item["uid"] = uid
    return out_item


def run_serial_data_test(client: Client, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for i, item in enumerate(questions):
        q = item.get("question", "")
        yaml_name = item.get("yaml", "adv_reason_3.yaml")
        if not yaml_name.endswith(".yaml"):
            yaml_name = f"{yaml_name}.yaml"
        submit_time = time.perf_counter()
        uid = client.submit(yaml_name, q)
        while True:
            st = client.status(uid)
            if st and st.get("status") == "completed":
                results.append(_build_result_item(item, uid, st, submit_time))
                break
            time.sleep(0.2)
        print(f"  serial progress: {i + 1}/{len(questions)}", flush=True)
    return results


def _run_one_mode(
    mode: str,
    questions: List[Dict[str, Any]],
    templates_abs: str,
    use_test_worker: bool,
    send_interval: float,
) -> List[Dict[str, Any]]:
    req_q = mp.Queue()
    resp_q = mp.Queue()
    proc = mp.Process(target=run_server, args=(req_q, resp_q, templates_abs, use_test_worker), daemon=False)
    proc.start()
    client = Client(req_q, resp_q)
    try:
        if mode == "parallel":
            results = run_data_test(client, questions, send_interval=send_interval)
        elif mode == "serial":
            results = run_serial_data_test(client, questions)
        else:
            raise ValueError(f"unknown mode: {mode}")
    finally:
        client.close()
        proc.join(timeout=5.0)
    _zero_timestamps(results)
    return _to_json_safe(results)


def main() -> None:
    p = argparse.ArgumentParser(description="Compare scheduler baseline: parallel vs serial")
    p.add_argument("--dataset", default="gsm8k", choices=("drop", "gsm8k", "hotpotqa", "math"))
    p.add_argument("-n", "--num", type=int, default=1000, help="number of questions")
    p.add_argument("--yaml", default="adv_reason_3.yaml", help="workflow yaml template")
    p.add_argument("--mode", default="both", choices=("parallel", "serial", "both"))
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--templates-dir", default="templates")
    p.add_argument("--send-interval", type=float, default=0.0)
    p.add_argument("--test-worker", action="store_true")
    p.add_argument("--worker-delay", type=float, default=None)
    p.add_argument("--output-dir", default="data/benchmarks")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    if args.verbose:
        set_verbose(True)
        os.environ["MFE_VERBOSE"] = "1"
    if args.worker_delay is not None:
        os.environ["MFE_TEST_WORKER_DELAY"] = str(args.worker_delay)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    templates_abs = os.path.abspath(args.templates_dir)
    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    questions = load_questions_from_parquet(args.dataset, os.path.join(root, "data"), args.yaml, args.num)
    if not questions:
        raise RuntimeError(f"No data for dataset={args.dataset}")

    modes = ["parallel", "serial"] if args.mode == "both" else [args.mode]
    summary: Dict[str, Any] = {
        "dataset": args.dataset,
        "num": len(questions),
        "yaml": args.yaml,
        "repeat": args.repeat,
        "test_worker": bool(args.test_worker),
        "modes": {},
    }

    for mode in modes:
        mode_runs: List[Dict[str, Any]] = []
        for run_idx in range(1, args.repeat + 1):
            print(f"[{mode}] run {run_idx}/{args.repeat} ...", flush=True)
            results = _run_one_mode(
                mode=mode,
                questions=questions,
                templates_abs=templates_abs,
                use_test_worker=args.test_worker,
                send_interval=args.send_interval,
            )
            stats = _calc_stats(results)
            mode_runs.append(stats)

            yaml_base = args.yaml.replace(".yaml", "")
            detail_name = f"{args.dataset}_{mode}_{yaml_base}_{len(questions)}_run{run_idx}.json"
            detail_path = os.path.join(out_dir, detail_name)
            with open(detail_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=_json_default)
            print(
                f"  saved detail: {detail_path}\n"
                f"  throughput={stats['throughput']:.4f} req/s, "
                f"lat_p95={stats['latency_p95']}",
                flush=True,
            )

        summary["modes"][mode] = mode_runs

    if "parallel" in summary["modes"] and "serial" in summary["modes"]:
        p_runs = summary["modes"]["parallel"]
        s_runs = summary["modes"]["serial"]
        p_tp = mean([r["throughput"] for r in p_runs]) if p_runs else 0.0
        s_tp = mean([r["throughput"] for r in s_runs]) if s_runs else 0.0
        tp_gain = ((p_tp - s_tp) / s_tp * 100.0) if s_tp > 0 else None
        p_p95 = mean([r["latency_p95"] for r in p_runs if r["latency_p95"] is not None]) if p_runs else None
        s_p95 = mean([r["latency_p95"] for r in s_runs if r["latency_p95"] is not None]) if s_runs else None
        summary["comparison"] = {
            "throughput_mean_parallel": p_tp,
            "throughput_mean_serial": s_tp,
            "throughput_gain_pct": tp_gain,
            "latency_p95_mean_parallel": p_p95,
            "latency_p95_mean_serial": s_p95,
        }

    summary_name = f"summary_{args.dataset}_{args.yaml.replace('.yaml', '')}_{len(questions)}.json"
    summary_path = os.path.join(out_dir, summary_name)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_to_json_safe(summary), f, ensure_ascii=False, indent=2, default=_json_default)
    print(f"saved summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
