"""
Workers 模块：导出 Worker 实现

该模块提供：
- vLLMWorker: 基于 vLLM 的 Worker，绑定 GPU，执行模型推理
- TestWorker: 测试用假 Worker，不依赖 vLLM/GPU，输出等于输入 prompt，用于流程验证
"""

from .worker_v import vLLMWorker
from .worker_test import TestWorker

__all__ = ["vLLMWorker", "TestWorker"]
