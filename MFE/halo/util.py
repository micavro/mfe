"""
Util 模块：提供通用的工具函数

该模块提供了一些通用的工具函数，主要用于：
- 数据类型转换（字符串到 torch.dtype）
- GPU 设备管理（解析可见 GPU ID）
"""

import torch
import os


def _resolve_dtype(dtype_spec):
    """
    将字符串类型的数据类型规范转换为 torch.dtype 对象
    
    支持多种常见的类型表示方式，提供灵活的输入格式。
    如果输入已经是 torch.dtype 对象，则直接返回。
    
    Args:
        dtype_spec: 数据类型规范，可以是字符串或 torch.dtype 对象
                    支持的字符串格式：
                    - "bfloat16" / "bf16"
                    - "float16" / "fp16" / "f16" / "half"
                    - "float32" / "fp32" / "f32" / "float"
    
    Returns:
        torch.dtype: 对应的 PyTorch 数据类型对象
                    如果无法识别，默认返回 torch.float32
    
    Example:
        >>> _resolve_dtype("bfloat16")
        torch.bfloat16
        >>> _resolve_dtype("fp16")
        torch.float16
        >>> _resolve_dtype(torch.float32)
        torch.float32
    """
    # 如果已经是 torch.dtype，直接返回
    if isinstance(dtype_spec, torch.dtype):
        return dtype_spec
    # 如果是字符串，查找映射表
    if isinstance(dtype_spec, str):
        key = dtype_spec.strip().lower()  # 转换为小写并去除空格
        table = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "f16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "f32": torch.float32,
            "float": torch.float32,
            "half": torch.float16,
        }
        return table.get(key, torch.float32)  # 如果找不到匹配，默认返回 float32
    # 其他情况，默认返回 float32
    return torch.float32


def _visible_physical_gpu_ids() -> list[int]:
    """
    解析当前可见的物理 GPU ID 列表，考虑 CUDA_VISIBLE_DEVICES 环境变量的影响
    
    该函数用于确定系统应该使用哪些 GPU。如果设置了 CUDA_VISIBLE_DEVICES 环境变量，
    则只返回该变量中指定的 GPU ID；否则返回所有可用的 GPU ID。
    
    Returns:
        list[int]: 物理 GPU ID 的整数列表
        
    Example:
        # 环境变量未设置，系统有 4 个 GPU
        >>> _visible_physical_gpu_ids()
        [0, 1, 2, 3]
        
        # 设置 CUDA_VISIBLE_DEVICES=0,2
        >>> _visible_physical_gpu_ids()
        [0, 2]
        
        # 设置 CUDA_VISIBLE_DEVICES=1
        >>> _visible_physical_gpu_ids()
        [1]
    
    Note:
        该函数在创建 Worker 进程时被调用，用于确定应该为哪些 GPU 创建 Worker
    """
    # 获取环境变量 CUDA_VISIBLE_DEVICES
    env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    # 如果未设置，返回所有可用 GPU 的 ID
    if not env:
        return list(range(torch.cuda.device_count()))
    # 如果已设置，解析逗号分隔的 ID 列表
    # 例如："0,2" -> [0, 2]
    return [int(x) for x in env.split(",") if x.strip() != ""]