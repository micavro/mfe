"""
Serve 模块：MFE 双 Queue 服务

提供 run_mfe_server：从 request_queue 取请求（id, prompt, template），
调用 OptimizerMFE 执行单请求完整 DAG，将响应（含 op_output、benchmark、total_answer_time）写入 response_queue。
"""

from halo.serve.mfe_server import run_mfe_server

__all__ = ["run_mfe_server"]
