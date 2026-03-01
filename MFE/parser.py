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

from halo.components import Operator, ModelConfig


# ---------------- Public API ---------------- #

def load_config(config_path: str) -> dict:
    """
    从文件路径加载 YAML 配置文件并解析为 Python 字典
    
    Args:
        config_path: YAML 配置文件的路径
        
    Returns:
        dict: 解析后的配置字典，包含 'ops'、'start_ops'、'end_ops' 等键
        
    Note:
        使用 yaml.safe_load 安全地解析 YAML，避免执行任意代码
    """
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
    # --- 验证必需字段：检查配置的完整性和正确性 ---
    conf_ops = config.get("ops")
    if not isinstance(conf_ops, dict) or not conf_ops:
        raise ValueError("Config must contain a non-empty 'ops' mapping.")
    start_keys = config.get("start_ops")
    end_keys = config.get("end_ops")
    if not isinstance(start_keys, list) or not start_keys:
        raise ValueError("'start_ops' must be a non-empty list of op ids.")
    if not isinstance(end_keys, list) or not end_keys:
        raise ValueError("'end_ops' must be a non-empty list of op ids.")

    # --- 创建 Operator 对象：为每个 OP ID 创建一个空的 Operator 对象 ---
    # 这些对象稍后会被填充依赖关系和配置信息
    ops: Dict[str, Operator] = {oid: Operator(id=oid) for oid in conf_ops.keys()}

    # --- 验证引用和必需字段：确保配置的正确性 ---
    for oid, spec in conf_ops.items():
        # 每个 OP 必须有 model 字段
        if "model" not in spec:
            raise ValueError(f"Op '{oid}' is missing required field 'model'.")
        # 获取输入输出依赖关系
        in_ids = spec.get("input_ops", []) or []   # 前驱节点 ID 列表
        out_ids = spec.get("output_ops", []) or []  # 后继节点 ID 列表
        # 验证引用的 OP ID 是否存在
        for rid in in_ids + out_ids:
            if rid not in ops:
                raise ValueError(f"Op '{oid}' references unknown op '{rid}' in inputs/outputs.")

    # --- 建立依赖关系、附加 ModelConfig、初始化运行时字段 ---
    models: Set[str] = set()  # 收集所有使用的模型名称
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
        # 1. 如果配置中显式指定了 keep_cache，则使用该值
        # 2. 否则，如果该 OP 的任意后继节点使用相同模型，则自动推断为 True
        #    这样可以优化性能：如果下游 OP 使用相同模型，保留 cache 可以避免重复计算前缀
        explicit_keep = spec.get("keep_cache", None)
        inferred_keep = any(conf_ops[oid2]["model"] == model for oid2 in output_ids)
        op.keep_cache = bool(explicit_keep) if explicit_keep is not None else inferred_keep

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
    """
    便捷函数：一次性完成配置加载和图构建
    
    Args:
        config_path: YAML 配置文件的路径
        
    Returns:
        Tuple[Dict[str, Operator], List[Operator], List[Operator], Set[str]]:
            - ops: op_id -> Operator 的字典
            - start_ops: 起始节点列表
            - end_ops: 终止节点列表
            - models: 所有使用的模型名称集合
            
    Note:
        这是 load_config 和 build_ops_from_config 的组合，简化调用
    """
    return build_ops_from_config(load_config(config_path))


# ---------------- Internals ---------------- #

def _compute_max_distances(ops: Dict[str, Operator], end_ops: List[Operator]) -> None:
    """
    为每个 OP 计算到任意终点 OP 的最长路径长度（边数），同时检测循环依赖
    
    使用深度优先搜索（DFS）和记忆化技术，时间复杂度 O(V + E)。
    如果检测到循环依赖，会抛出 ValueError。
    
    Args:
        ops: 所有 Operator 的字典
        end_ops: 终点节点列表
        
    Raises:
        ValueError: 如果检测到图中存在循环依赖
        
    Note:
        - 如果 OP 无法到达任何终点，其距离为 -1
        - max_distance 字段用于调度优化：距离终点越远的节点通常应该优先执行
    """
    end_set = set(end_ops)  # 终点节点集合，用于快速查找
    memo: Dict[Operator, int] = {}  # 记忆化：避免重复计算
    visiting: Set[Operator] = set()  # 当前访问路径，用于循环检测

    def dfs(op: Operator) -> int:
        """
        深度优先搜索：计算 op 到任意终点的最长距离
        
        Returns:
            int: 最长距离（边数），如果无法到达终点则返回 -1
        """
        # 记忆化：如果已经计算过，直接返回
        if op in memo:
            return memo[op]
        # 循环检测：如果访问到正在访问的节点，说明存在环
        if op in visiting:
            raise ValueError("Cycle detected in the op graph.")
        visiting.add(op)  # 标记为正在访问

        # 如果当前节点是终点，距离为 0
        if op in end_set:
            memo[op] = 0
            visiting.remove(op)
            return 0

        # 递归计算所有后继节点的距离，取最大值
        best = -1  # -1 表示无法到达终点
        for child in op.output_ops:
            d = dfs(child)
            if d != -1:  # 如果后继节点可以到达终点
                best = max(best, d + 1)  # 当前节点距离 = 后继节点距离 + 1

        memo[op] = best  # 记录结果
        visiting.remove(op)  # 移除访问标记
        return best

    # 为所有 OP 计算距离并写入 max_distance 字段
    for op in ops.values():
        op.max_distance = dfs(op)
