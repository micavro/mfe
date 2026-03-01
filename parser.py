"""从 YAML 构建 Operator DAG。配置含 ops、start_ops、end_ops；每 op 含 model、input_ops、output_ops 等。"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple
import yaml
import torch

from mfe.components import Operator, ModelConfig


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_ops_from_config(config: dict) -> Tuple[Dict[str, Operator], List[Operator], List[Operator], Set[str]]:
    """从 config 构建 DAG，填 op 的 input_ops/output_ops/model_config 及 max_distance。返回 (ops, start_ops, end_ops, models)。"""
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
        input_ids = spec.get("input_ops", []) or []
        output_ids = spec.get("output_ops", []) or []
        op.input_ops = [ops[k] for k in input_ids]
        op.output_ops = [ops[k] for k in output_ids]
        model = spec["model"]
        models.add(model)
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
        op.data_parallel = False
        op.is_duplicate = False
        op.duplicate_info = None
        op.main_op = None
        op.max_distance = -1

    for k in start_keys + end_keys:
        if k not in ops:
            raise ValueError(f"Unknown op id '{k}' referenced in start_ops/end_ops.")
    start_ops = [ops[k] for k in start_keys]
    end_ops = [ops[k] for k in end_keys]
    _compute_max_distances(ops, end_ops)
    return ops, start_ops, end_ops, models


def build_from_path(config_path: str) -> Tuple[Dict[str, Operator], List[Operator], List[Operator], Set[str]]:
    return build_ops_from_config(load_config(config_path))


def _compute_max_distances(ops: Dict[str, Operator], end_ops: List[Operator]) -> None:
    """DFS 计算每 op 到任意 end_op 的最长距离，遇环抛错。"""
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
