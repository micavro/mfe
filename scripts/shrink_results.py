#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将 results JSON 精简为只保留 answer 和 gold_answer。
先读取 JSON，处理非法 UTF-8 字符（直接忽略），写入临时文件，再从该文件解析并精简。"""

from __future__ import annotations

import argparse
import io
import json
import os


def main() -> None:
    p = argparse.ArgumentParser(description="精简 results JSON，只保留 answer 和 gold_answer")
    p.add_argument("input", help="输入 JSON 路径，如 data/results_hotpotqa.json")
    p.add_argument("-o", "--output", default=None, help="输出路径，默认 inp_compact.json")
    p.add_argument("--keep-fixed", action="store_true", help="保留 UTF-8 修复后的中间文件")
    args = p.parse_args()

    inp = os.path.abspath(args.input)
    if not os.path.isfile(inp):
        print(f"File not found: {inp}")
        return

    out = args.output
    if not out:
        base, ext = os.path.splitext(inp)
        out = f"{base}_compact{ext}"

    base, ext = os.path.splitext(inp)
    fixed_path = f"{base}_utf8fix{ext}"

    try:
        from tqdm import tqdm
    except ImportError:
        class _DummyTqdm:
            def __init__(self, iterable=None, **kw):
                self._it = iter(iterable) if iterable is not None else None
            def __iter__(self):
                return iter(self._it) if self._it else iter([])
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass
            def update(self, n=1):
                pass
        tqdm = _DummyTqdm

    size = os.path.getsize(inp)
    print(f"Step 1: Reading {inp} ({size / 1024 / 1024:.1f} MB)...")
    chunks = []
    read_size = 1024 * 1024
    with open(inp, "rb") as f:
        with tqdm(total=size, unit="B", unit_scale=True, unit_divisor=1024, desc="Read") as pbar:
            while True:
                chunk = f.read(read_size)
                if not chunk:
                    break
                chunks.append(chunk)
                pbar.update(len(chunk))
    raw = b"".join(chunks)

    print("Processing illegal UTF-8 (ignore invalid bytes)...")
    text = raw.decode("utf-8", errors="ignore")
    # JSON 不允许字符串内出现未转义的控制字符 (ASCII 0-31)，全部替换为空格
    print("Stripping control characters...")
    text = text.translate({i: ord(" ") for i in range(32)})
    print(f"Writing sanitized file to {fixed_path}...")
    with open(fixed_path, "w", encoding="utf-8") as f:
        f.write(text)

    compact = []
    use_ijson = False
    try:
        import ijson
        use_ijson = True
    except ImportError:
        pass

    print(f"Step 2: Parsing {fixed_path}...")
    if use_ijson:
        try:
            with open(fixed_path, "rb") as f:
                for item in tqdm(ijson.items(f, "item"), desc="Parse", unit=" items"):
                    compact.append({
                        "answer": str(item.get("answer", "")),
                        "gold_answer": str(item.get("gold_answer", "")),
                    })
        except Exception as e:
            print(f"  ijson failed ({e}), fallback to stdlib json")
            use_ijson = False
            compact = []

    if not use_ijson:
        with open(fixed_path, "r", encoding="utf-8") as f:
            text_to_parse = f.read()
        try:
            data = json.loads(text_to_parse)
        except json.JSONDecodeError:
            try:
                import json_repair
                data = json_repair.loads(text_to_parse)
                print("  Used json_repair to fix malformed JSON")
            except ImportError:
                print("  pip install json-repair 可尝试修复含非法控制字符的 JSON")
                raise
        if not isinstance(data, list):
            data = [data]
        for item in tqdm(data, desc="Shrink", unit=" items"):
            compact.append({
                "answer": str(item.get("answer", "")),
                "gold_answer": str(item.get("gold_answer", "")),
            })

    with open(out, "w", encoding="utf-8") as f:
        json.dump(compact, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(compact)} items to {out}")
    if not args.keep_fixed:
        os.remove(fixed_path)
        print(f"Removed temp file {fixed_path}")
    else:
        print(f"Kept temp file {fixed_path}")


if __name__ == "__main__":
    main()
