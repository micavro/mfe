"""run_server：多请求异步；run_mfe_server：单请求双队列（需 halo）。"""

from .server import run_server
try:
    from halo.serve.mfe_server import run_mfe_server
except ImportError:
    run_mfe_server = None
__all__ = ["run_server", "run_mfe_server"]
