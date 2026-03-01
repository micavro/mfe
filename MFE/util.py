"""工具：dtype 解析、可见 GPU ID 列表。"""

import os
import torch

def _resolve_dtype(dtype_spec):
    """字符串或 torch.dtype → torch.dtype。支持 bf16/fp16/fp32 等。"""
    if isinstance(dtype_spec, torch.dtype):
        return dtype_spec
    if isinstance(dtype_spec, str):
        key = dtype_spec.strip().lower()
        table = {
            "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
            "fp16": torch.float16, "float16": torch.float16, "f16": torch.float16, "half": torch.float16,
            "float32": torch.float32, "fp32": torch.float32, "f32": torch.float32, "float": torch.float32,
        }
        return table.get(key, torch.float32)
    return torch.float32


def _visible_physical_gpu_ids() -> list[int]:
    """解析 CUDA_VISIBLE_DEVICES，返回物理 GPU ID 列表。未设置则返回全部。"""
    env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not env:
        return list(range(torch.cuda.device_count()))
    return [int(x) for x in env.split(",") if x.strip()]
