"""全局配置：verbose 等。通过 set_verbose() 在入口设置。"""

from __future__ import annotations

_verbose: bool = True


def is_verbose() -> bool:
    """是否输出 verbose 日志。"""
    return _verbose


def set_verbose(verbose: bool) -> None:
    """设置 verbose。建议在程序入口调用。"""
    global _verbose
    _verbose = bool(verbose)
