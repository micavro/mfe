"""全局配置：verbose 等。优先级：环境变量 MFE_VERBOSE > mfe_config.yaml。"""

from __future__ import annotations

import os
from typing import Any, Dict

_config: Dict[str, Any] = {}


def _load_config() -> Dict[str, Any]:
    global _config
    if _config:
        return _config
    cfg = {"verbose": False}
    for path in [
        os.path.join(os.getcwd(), "mfe_config.yaml"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "mfe_config.yaml"),
    ]:
        if os.path.isfile(path):
            try:
                import yaml
                with open(path, "r") as f:
                    data = yaml.safe_load(f) or {}
                cfg["verbose"] = bool(data.get("verbose", False))
            except Exception:
                pass
            break
    if os.environ.get("MFE_VERBOSE", "").lower() in ("1", "true", "yes"):
        cfg["verbose"] = True
    _config = cfg
    return _config


def is_verbose() -> bool:
    """是否输出 verbose 日志。"""
    return _load_config().get("verbose", False)


def set_verbose(verbose: bool) -> None:
    """设置 verbose（写环境变量，子进程继承）。"""
    os.environ["MFE_VERBOSE"] = "1" if verbose else "0"
    global _config
    _config.clear()
