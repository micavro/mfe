"""
Serve 模块：MFE 服务

- run_server: 多请求异步 Server (submit/status API)
- run_mfe_server: 单请求双 Queue 服务（需 halo）
"""

from .server import run_server

try:
    from halo.serve.mfe_server import run_mfe_server
except ImportError:
    run_mfe_server = None

__all__ = ["run_server", "run_mfe_server"]
