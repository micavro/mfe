"""轮询调度：拓扑序将 OP 轮询分配到各设备。"""

from typing import List, Dict
from collections import deque
from mfe.components import Operator, Query


def _topo_order(all_ops: List[Operator]) -> List[Operator]:
    """Kahn 拓扑排序。非 DAG 抛 ValueError。"""
    indeg = {op: len(op.input_ops) for op in all_ops}
    q = deque([op for op, d in indeg.items() if d == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in u.output_ops:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(topo) != len(all_ops):
        raise ValueError("Graph is not a DAG.")
    return topo


def schedule_rr(device_cnt: int, all_ops: List[Operator], queries: List[Query]) -> List[List[Dict]]:
    """按拓扑序轮询分配 OP 到 device_cnt 个设备。每项 {"command": "execute", "params": (op, query_ids)}。"""
    if device_cnt <= 0:
        raise RuntimeError("No devices available.")
    workflows = [[] for _ in range(device_cnt)]
    all_ids = [q.id for q in queries]
    topo = _topo_order(all_ops)
    d = 0
    for op in topo:
        workflows[d].append({"command": "execute", "params": (op, list(all_ids))})
        d = (d + 1) % device_cnt
    return workflows
