"""全局配置：verbose 等。通过 set_verbose() 或环境变量 MFE_VERBOSE 设置。子进程继承环境变量。"""

from __future__ import annotations

import os

_verbose: bool = False


def is_verbose() -> bool:
    """是否输出 verbose 日志。环境变量 MFE_VERBOSE 优先，便于子进程继承。"""
    if os.environ.get("MFE_VERBOSE", "").lower() in ("1", "true", "yes"):
        return True
    return _verbose


def set_verbose(verbose: bool) -> None:
    """设置 verbose。建议在程序入口调用。若需子进程（server/worker）也 verbose，同时设置 MFE_VERBOSE=1。"""
    global _verbose
    _verbose = bool(verbose)
