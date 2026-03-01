"""
Schedulers 模块：将工作流图中的 Operator 分配到多设备执行

当前仅提供轮询调度 schedule_rr：按拓扑序将每个 OP 轮询分配到各 GPU，每 OP 处理当前全部查询。
"""

from .rr import schedule_rr

__all__ = ["schedule_rr"]
