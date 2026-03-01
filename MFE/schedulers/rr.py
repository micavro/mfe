"""
轮询调度器（Round-Robin Scheduler）

该模块实现了最简单的调度策略：按照拓扑顺序，将每个 OP 轮询分配到设备上。
每个 OP 处理所有查询，不考虑模型切换成本、缓存局部性等因素。

适用场景：
- 基准测试和对比
- 简单的工作流（OP 数量少，依赖关系简单）
- 所有 OP 使用相同模型的情况
"""

from typing import List, Dict
from collections import deque
from mfe.components import Operator, Query


def _topo_order(all_ops: List[Operator]) -> List[Operator]:
    """
    拓扑排序：使用 Kahn 算法对 OP 进行拓扑排序
    
    返回一个满足依赖关系的 OP 执行顺序，确保父节点总是先于子节点。
    这是多个调度器共享的工具函数。
    
    Args:
        all_ops: 所有 Operator 的列表
        
    Returns:
        List[Operator]: 拓扑排序后的 OP 列表
        
    Raises:
        ValueError: 如果图中存在循环（不是 DAG）
        
    Algorithm:
        Kahn 算法（基于入度的拓扑排序）
        1. 计算每个节点的入度（input_ops 的数量）
        2. 将所有入度为 0 的节点加入队列
        3. 从队列中取出节点，加入结果列表
        4. 将该节点的所有后继节点的入度减 1
        5. 如果某个后继节点的入度变为 0，将其加入队列
        6. 重复步骤 3-5，直到队列为空
        
    Time Complexity: O(V + E)，其中 V 是节点数，E 是边数
    """
    # 计算每个节点的入度（前驱节点数量）
    indeg = {op: len(op.input_ops) for op in all_ops}
    # 将所有入度为 0 的节点（起始节点）加入队列
    q = deque([op for op, d in indeg.items() if d == 0])
    topo = []  # 拓扑排序结果
    
    while q:
        u = q.popleft()  # 取出一个入度为 0 的节点
        topo.append(u)   # 加入结果列表
        
        # 遍历该节点的所有后继节点
        for v in u.output_ops:
            indeg[v] -= 1  # 后继节点的入度减 1
            if indeg[v] == 0:  # 如果入度变为 0，加入队列
                q.append(v)
    
    # 如果排序后的节点数不等于总节点数，说明存在循环
    if len(topo) != len(all_ops):
        raise ValueError("Graph is not a DAG.")
    return topo


def schedule_rr(device_cnt: int, all_ops: List[Operator], queries: List[Query]) -> List[List[Dict]]:
    """
    轮询调度：按照拓扑顺序，将每个 OP 轮询分配到设备上
    
    最简单的调度策略，不考虑模型切换成本、缓存局部性等因素。
    每个 OP 处理所有查询，设备之间负载均衡。
    
    Args:
        device_cnt: 可用设备（GPU）数量
        all_ops: 所有 Operator 的列表
        queries: 所有查询的列表
        
    Returns:
        List[List[Dict]]: 每个设备上的执行计划（workflow）
                         格式：List[ per-device List[ {"command": "execute", "params": (op, query_ids)} ] ]
        
    Raises:
        RuntimeError: 如果设备数量 <= 0
        
    Strategy:
        1. 对所有 OP 进行拓扑排序
        2. 按顺序遍历每个 OP
        3. 将 OP 分配到设备 d，并将所有查询 ID 传递给该 OP
        4. 设备索引循环递增：d = (d + 1) % device_cnt
        
    Example:
        假设有 3 个设备，4 个 OP（op0 -> op1 -> op2 -> op3）：
        - 设备 0: [op0, op3]
        - 设备 1: [op1]
        - 设备 2: [op2]
    """
    if device_cnt <= 0:
        raise RuntimeError("No devices available.")

    # 初始化每个设备的工作流列表
    workflows: List[List[Dict]] = [[] for _ in range(device_cnt)]
    # 提取所有查询的 ID
    all_ids = [q.id for q in queries]
    # 对 OP 进行拓扑排序，确保依赖关系正确
    topo = _topo_order(all_ops)

    # 轮询分配：按拓扑顺序将每个 OP 分配到设备
    d = 0  # 当前设备索引
    for op in topo:
        # 将 OP 分配到设备 d，处理所有查询
        workflows[d].append({"command": "execute", "params": (op, list(all_ids))})
        # 设备索引循环递增
        d = (d + 1) % device_cnt
    return workflows
