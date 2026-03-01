"""
parser.py
Build a DAG of Operator objects from a YAML config (ops/* only).

Config schema (no legacy node keys):
- ops:
    <op_id>:
        model: <str>
        input_ops: [<op_id>, ...]     # optional
        output_ops: [<op_id>, ...]    # optional
        prompt: <str>                 # optional
        temperature: <float>          # default 0.7
        top_p: <float>                # default 0.9
        max_tokens: <int>             # default 256
        max_batch_size: <int|inf>     # default torch.inf
        dtype: <"bfloat16"|"float16"|...>    # default "bfloat16"
        quantization: <any>           # optional
        lora_config: <dict|None>      # optional
        max_model_len: <int|None>     # optional
        min_tokens: <int>             # default 0
        use_chat_template: <bool>     # default False
        keep_cache: <bool>            # optional; overrides inference
- start_ops: [<op_id>, ...]
- end_ops:   [<op_id>, ...]

During build we also:
- infer keep_cache if any downstream op shares the same model (unless keep_cache is explicitly set)
- initialize runtime fields: data_parallel, is_duplicate, duplicate_info, main_op, benchmark, max_distance
- compute max_distance to any end-op (with cycle detection)
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple
import yaml
import torch

from mfe.components import Operator, ModelConfig


# ---------------- Public API ---------------- #

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_ops_from_config(config: dict) -> Tuple[Dict[str, Operator], List[Operator], List[Operator], Set[str]]:
    """
    Build Operator graph from a config dict (ops/* only) and compute max_distance per op.

    Returns
    -------
    ops        : dict[str, Operator]
    start_ops  : list[Operator]
    end_ops    : list[Operator]
    models     : set[str]
    """
    conf_ops = config.get("ops")
    if not isinstance(conf_ops, dict) or not conf_ops:
        raise ValueError("Config must contain a non-empty 'ops' mapping.")
    start_keys = config.get("start_ops")
    end_keys = config.get("end_ops")
    if not isinstance(start_keys, list) or not start_keys:
        raise ValueError("'start_ops' must be a non-empty list of op ids.")
    if not isinstance(end_keys, list) or not end_keys:
        raise ValueError("'end_ops' must be a non-empty list of op ids.")

    ops: Dict[str, Operator] = {oid: Operator(id=oid) for oid in conf_ops.keys()}

    for oid, spec in conf_ops.items():
        if "model" not in spec:
            raise ValueError(f"Op '{oid}' is missing required field 'model'.")
        in_ids = spec.get("input_ops", []) or []
        out_ids = spec.get("output_ops", []) or []
        for rid in in_ids + out_ids:
            if rid not in ops:
                raise ValueError(f"Op '{oid}' references unknown op '{rid}' in inputs/outputs.")

    models: Set[str] = set()
    for oid, op in ops.items():
        spec = conf_ops[oid]
        input_ids = spec.get("input_ops", []) or []   # 前驱节点 ID 列表
        output_ids = spec.get("output_ops", []) or []  # 后继节点 ID 列表

        # 建立双向链接：将字符串 ID 转换为 Operator 对象引用
        op.input_ops = [ops[k] for k in input_ids]    # 前驱节点对象列表
        op.output_ops = [ops[k] for k in output_ids]  # 后继节点对象列表

        # 收集模型名称
        model = spec["model"]
        models.add(model)

        # keep_cache 推断逻辑：

        op.model_config = ModelConfig(
            model_name=model,
            system_prompt=spec.get("prompt"),
            temperature=spec.get("temperature", 0.7),
            top_p=spec.get("top_p", 0.9),
            max_tokens=spec.get("max_tokens", 256),
            max_batch_size=spec.get("max_batch_size", torch.inf),
            dtype=spec.get("dtype", "bfloat16"),
            quantization=spec.get("quantization", None),
            lora_config=spec.get("lora_config", None),
            max_model_len=spec.get("max_model_len", None),
            min_tokens=spec.get("min_tokens", 0),
            use_chat_template=spec.get("use_chat_template", True),
        )

        # 初始化运行时/调度器字段（这些字段在调度阶段会被修改）
        op.data_parallel = False      # 是否采用数据并行（多设备复制执行）
        op.is_duplicate = False        # 是否为复制节点（用于数据并行）
        op.duplicate_info = None       # 复制信息 [dup_index, total_dup]，由调度器设置
        op.main_op = None              # 如果为复制节点，指向原始主节点
        op.max_distance = -1           # 到终点的最长距离，稍后由 _compute_max_distances 计算

    # --- 解析 start_ops 和 end_ops：将字符串 ID 转换为 Operator 对象列表 ---
    for k in start_keys + end_keys:
        if k not in ops:
            raise ValueError(f"Unknown op id '{k}' referenced in start_ops/end_ops.")
    start_ops = [ops[k] for k in start_keys]  # 起始节点对象列表
    end_ops = [ops[k] for k in end_keys]      # 终止节点对象列表

    # --- 计算最长距离到任意终点节点（同时检测循环依赖） ---
    # max_distance 字段用于调度优化：距离终点越远的节点通常应该优先执行
    _compute_max_distances(ops, end_ops)

    return ops, start_ops, end_ops, models


def build_from_path(config_path: str) -> Tuple[Dict[str, Operator], List[Operator], List[Operator], Set[str]]:
    return build_ops_from_config(load_config(config_path))


# ---------------- Internals ---------------- #

def _compute_max_distances(ops: Dict[str, Operator], end_ops: List[Operator]) -> None:
    end_set = set(end_ops)
    memo: Dict[Operator, int] = {}
    visiting: Set[Operator] = set()

    def dfs(op: Operator) -> int:
        if op in memo:
            return memo[op]
        if op in visiting:
            raise ValueError("Cycle detected in the op graph.")
        visiting.add(op)

        if op in end_set:
            memo[op] = 0
            visiting.remove(op)
            return 0

        best = -1
        for child in op.output_ops:
            d = dfs(child)
            if d != -1:
                best = max(best, d + 1) 

        memo[op] = best
        visiting.remove(op)
        return best

    for op in ops.values():
        op.max_distance = dfs(op)
