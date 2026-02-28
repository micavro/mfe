#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MFE 测试脚本：按计划向 Server 发送请求，收集结果并统计

- 数据集由命令行指定（--dataset/--name/--column），从 HuggingFace 下载后保存为 JSONL 到 --data-dir，后续优先从 JSONL 读取。
- 请求为 [prompt, template]，template 为工作流 YAML 文件名。
- 主进程创建 request_queue / response_queue，启动 MFE Server 子进程，顺序发送请求并收取响应。
- 统计：成功数、失败数、总答案时间 P50/P95、可选每节点耗时。
"""

from __future__ import annotations

import os
import warnings

# 避免多份 OpenMP 运行时冲突（常见于 Windows 下 PyTorch/NumPy 等混用）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# 抑制 torch.cuda 对 pynvml 的弃用提示，避免刷屏
warnings.filterwarnings("ignore", category=FutureWarning, message=".*pynvml.*")

import json
import re
import sys
import uuid
import argparse
import multiprocessing as mp
from typing import List, Dict, Any

import numpy as np
from datasets import load_dataset, concatenate_datasets, Dataset

# 保证项目根在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from halo.serve.mfe_server import run_mfe_server


# ---------------------------------------------------------------------------
# 数据集路径与 JSONL 读写
# ---------------------------------------------------------------------------

def _dataset_jsonl_basename(dataset: str, name: str, column: str) -> str:
    """生成数据集对应的 JSONL 文件名（不含路径）。"""
    safe = re.sub(r"[^\w\-]", "_", f"{dataset}_{name}_{column}")
    return f"{safe}.jsonl"


def get_dataset_jsonl_path(data_dir: str, dataset: str, name: str, column: str) -> str:
    """返回数据集对应的 JSONL 文件完整路径。"""
    return os.path.join(data_dir, _dataset_jsonl_basename(dataset, name, column))


def load_dataset_from_hf(
    dataset: str,
    name: str | None,
    column: str,
    split: str = "train",
) -> List[Dict[str, Any]]:
    """
    从 HuggingFace 加载数据集，返回行字典列表。
    name 为空时使用默认 config。
    """
    ds = load_dataset(dataset, name=name, split=split)
    out = []
    for i in range(len(ds)):
        row = ds[i]
        if isinstance(row, dict):
            out.append(dict(row))
        else:
            out.append({column: str(row)})
    return out


def save_dataset_jsonl(samples: List[Dict[str, Any]], path: str) -> None:
    """将样本列表保存为 JSONL，每行一个 JSON 对象。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in samples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_dataset_jsonl(path: str) -> List[Dict[str, Any]]:
    """从 JSONL 文件加载样本列表。"""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def ensure_dataset(
    data_dir: str,
    dataset: str,
    name: str | None,
    column: str,
    num_requests: int,
    split: str = "train",
) -> List[Dict[str, Any]]:
    """
    获取用于测试的样本列表：若 data_dir 下已有对应 JSONL 则从中读取（取前 num_requests 条）；
    否则从 HuggingFace 下载，保存为 JSONL，再返回前 num_requests 条。
    """
    path = get_dataset_jsonl_path(data_dir, dataset, name or "default", column)
    if os.path.isfile(path):
        samples = load_dataset_jsonl(path)
    else:
        samples = load_dataset_from_hf(dataset, name, column, split=split)
        if num_requests > len(samples):
            times = num_requests // len(samples) + 1
            ds = Dataset.from_list(samples)
            ds = concatenate_datasets([ds] * times)
            samples = [ds[i] for i in range(min(num_requests, len(ds)))]
        save_dataset_jsonl(samples, path)
    # num_requests<=0 表示使用全部样本；否则取前 num_requests 条
    if num_requests is None or num_requests <= 0:
        return samples
    return samples[:num_requests]


def build_mfe_requests(
    samples: List[Dict[str, Any]],
    text_key: str,
    template_name: str,
    max_input_len: int,
) -> List[Dict[str, Any]]:
    """
    构建 MFE 请求列表：{"id", "prompt", "template"}。
    若某行带有 "template" 字段，则用该值作为该询问的 YAML；否则用默认 template_name（来自 --template）。
    """
    requests = []
    for i, row in enumerate(samples):
        raw = row.get(text_key, "")
        text = raw if isinstance(raw, str) else str(raw)
        if max_input_len and max_input_len > 0:
            text = text[:max_input_len]
        # 每行可指定不同工作流：row["template"] 存在则用，否则用默认（均为 YAML 文件名）
        tpl = (row.get("template") or template_name).strip() or template_name
        if tpl and not tpl.endswith(".yaml"):
            tpl = tpl + ".yaml"
        tpl = os.path.basename(tpl)  # 只传文件名，不含路径
        requests.append({
            "id": str(uuid.uuid4()),
            "prompt": text,
            "template": tpl,
        })
    return requests


def main() -> None:
    parser = argparse.ArgumentParser(description="MFE 测试：双 Queue 单请求执行，统计总答案时间")
    parser.add_argument(
        "--template",
        type=str,
        default="templates/adv_reason_3.yaml",
        help="默认工作流 YAML 文件名；若数据行中有 template 字段则按行覆盖",
    )
    parser.add_argument(
        "--templates-dir",
        type=str,
        default="templates",
        help="工作流 YAML 根目录（与 Server 一致）",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="数据集 JSONL 存放目录（不存在则创建）；优先从该目录读取 JSONL",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="openai/gsm8k",
        help="HuggingFace 数据集名（用于下载并保存为 JSONL）",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="main",
        help="HuggingFace 数据集 config 名（可选）",
    )
    parser.add_argument(
        "--column",
        type=str,
        default="question",
        help="用作 prompt 的数据集列名",
    )
    parser.add_argument(
        "-n", "--num_queries",
        type=int,
        default=0,
        help="请求数量；0 表示使用数据集中全部样本",
    )
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=256,
        help="单条 prompt 最大长度",
    )
    parser.add_argument(
        "--test-worker",
        action="store_true",
        help="使用假 Worker（不依赖 vLLM/GPU，输出=输入），仅用于验证流程",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="打印每条请求/响应、每个 Worker 收发、询问经过的节点与时间等中间信息",
    )
    args = parser.parse_args()

    if args.verbose:
        os.environ["MFE_VERBOSE"] = "1"

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    data_dir = os.path.abspath(args.data_dir)
    template_path = args.template
    template_name = os.path.basename(template_path)

    samples = ensure_dataset(
        data_dir,
        args.dataset,
        args.name or None,
        args.column,
        args.num_queries,
    )
    requests = build_mfe_requests(
        samples, args.column, template_name, args.max_input_length
    )

    request_queue = mp.Queue()
    response_queue = mp.Queue()
    templates_dir_abs = os.path.abspath(args.templates_dir)

    server_proc = mp.Process(
        target=run_mfe_server,
        args=(request_queue, response_queue, templates_dir_abs, args.test_worker, args.verbose),
        daemon=False,
    )
    server_proc.start()

    latencies: List[float] = []
    errors: List[str] = []
    node_elapsed: Dict[str, List[float]] = {}

    for req in requests:
        if args.verbose:
            pid = str(req.get("id", ""))[:8]
            tpl = req.get("template", "")
            prompt_preview = (req.get("prompt", "") or "")[:70]
            if len(req.get("prompt", "") or "") > 70:
                prompt_preview += "..."
            print(f"[REQ] id={pid} template={tpl} prompt={prompt_preview!r}")
        request_queue.put(req)
        resp = response_queue.get()
        if resp.get("ok"):
            t = resp["result"].get("total_answer_time")
            if t is not None:
                latencies.append(t)
            bench = resp["result"].get("benchmark") or {}
            for op_id, (start, end) in bench.items():
                node_elapsed.setdefault(op_id, []).append(end - start)
            if args.verbose:
                rid = str(resp.get("id", ""))[:8]
                total = resp["result"].get("total_answer_time")
                # 按节点开始时间排序，得到询问经过的链路
                path_parts = []
                for op_id, (start, end) in sorted(bench.items(), key=lambda x: x[1][0]):
                    path_parts.append(f"{op_id}({start:.3f}-{end:.3f})")
                path_str = " -> ".join(path_parts)
                out_keys = list((resp["result"].get("op_output") or {}).keys())
                print(f"[RSP] id={rid} ok total_answer_time={total:.3f}s path=[ {path_str} ] op_output_keys={out_keys}")
        else:
            errors.append(resp.get("error", "unknown"))
            if args.verbose:
                print(f"[RSP] id={str(resp.get('id',''))[:8]} ok=False error={resp.get('error', 'unknown')}")

    request_queue.put(None)
    server_proc.join()

    print("[MFE Test] Done.")
    print(f"  Total: {len(requests)}, OK: {len(latencies)}, Failed: {len(errors)}")
    if errors:
        for e in errors[:5]:
            print(f"  Error: {e}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")
    if latencies:
        arr = np.array(latencies)
        print(f"  Total answer time (s): P50={np.percentile(arr, 50):.3f}, P95={np.percentile(arr, 95):.3f}")
        for op_id, times in sorted(node_elapsed.items()):
            a = np.array(times)
            print(f"  Node {op_id} elapsed (s): P50={np.percentile(a, 50):.3f}, P95={np.percentile(a, 95):.3f}")


if __name__ == "__main__":
    main()
