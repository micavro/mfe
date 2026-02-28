"""
TestWorker 模块：用于流程测试的假 Worker

不依赖 vLLM/GPU，仅将输入 prompt 原样返回为 output，用于在本机无法运行 vLLM 时验证
MFE 端到端流程（Server → Optimizer → Worker → 结果与性能统计）。
与 vLLMWorker 保持相同的消息格式与 execute 返回结构，可直接替换使用。
"""

from __future__ import annotations

import os
import time
import logging
from typing import Any, Dict, List

from halo.components import ExecuteInfo

logger = logging.getLogger(__name__)


class TestWorker:
    """
    测试用 Worker：不加载模型，execute 时直接返回输入 prompt 作为 output。

    接口与 vLLMWorker 一致：
    - __init__(id, physical_gpu_id, cmd_queue, result_queue)
    - run(debug=...)
    - execute(exe_info) -> {"item": [...], "op_name": str, "benchmark": {...}}
    - exit() -> str

    返回给主进程的消息格式：{"command": str, "result": Any, "elapsed_time": float}
    """

    def __init__(
        self,
        id: int,
        physical_gpu_id: int,
        cmd_queue: "Any",
        result_queue: "Any",
    ) -> None:
        self.id = id
        self.physical_gpu_id = physical_gpu_id
        self.cmd_queue = cmd_queue
        self.response_queue = result_queue
        logger.info("TestWorker[%s] initialized (no GPU, echo mode)", id)

    def execute(self, exe_info: ExecuteInfo) -> Dict[str, Any]:
        """
        直接返回每个 prompt 作为对应 output，并构造与 vLLMWorker 一致的返回结构。
        """
        op = exe_info.op
        op_name = getattr(op, "id", "op_unknown")
        is_duplicate = getattr(op, "is_duplicate", False)

        start = time.perf_counter()
        results: List[Dict[str, Any]] = []
        for qid, prompt in zip(exe_info.query_ids, exe_info.prompts):
            # 输出 = 输入 prompt（便于验证流程）
            text = prompt if isinstance(prompt, str) else str(prompt)
            end = time.perf_counter()
            results.append({
                "id": qid,
                "output": text,
                "benchmark": (start, end),
            })
        elapsed = time.perf_counter() - start

        if is_duplicate:
            benchmark = {"init_time": 0.0, "prefill_time": 0.0, "generate_time": 0.0}
        else:
            benchmark = {
                "init_time": 0.0,
                "prefill_time": 0.0,
                "generate_time": elapsed,
            }
        return {"item": results, "op_name": op_name, "benchmark": benchmark}

    def exit(self) -> str:
        logger.info("TestWorker[%s] exited.", self.id)
        return "TestWorker exited."

    def run(self, debug: bool = True) -> None:
        """与 vLLMWorker 相同的主循环：从 cmd_queue 取命令，执行后写入 result_queue。"""
        while True:
            msg = self.cmd_queue.get()

            if isinstance(msg, tuple):
                command, params = msg
            elif isinstance(msg, dict):
                command = msg.get("command")
                params = msg.get("params", ())
            else:
                self.response_queue.put(
                    {"command": "error", "result": "Unsupported message format", "elapsed_time": 0.0}
                )
                continue

            if command == "exit":
                out = self.exit()
                self.response_queue.put({"command": "exit", "result": out, "elapsed_time": 0.0})
                break

            _verbose = os.environ.get("MFE_VERBOSE", "").lower() in ("1", "true", "yes")
            if _verbose and command == "execute" and params:
                exe = params[0]
                op_id = getattr(exe.op, "id", "?")
                qids = getattr(exe, "query_ids", [])
                prompts = getattr(exe, "prompts", [])
                p0 = (prompts[0][:50] + "...") if prompts and len(prompts[0]) > 50 else (prompts[0] if prompts else "")
                print(f"[Worker {self.id}] recv execute op={op_id} query_ids={qids} n_prompts={len(prompts)} prompt0={p0!r}")

            func = getattr(self, command, None)
            if not callable(func):
                self.response_queue.put(
                    {"command": "error", "result": f"Unknown command: {command}", "elapsed_time": 0.0}
                )
                continue

            start = time.perf_counter()
            try:
                result = func(*params) if params else func()
                elapsed = time.perf_counter() - start
                self.response_queue.put({"command": command, "result": result, "elapsed_time": elapsed})
                if _verbose and command == "execute" and isinstance(result, dict):
                    print(f"[Worker {self.id}] sent result op_name={result.get('op_name', '?')} elapsed={elapsed:.3f}s")
            except Exception as e:
                if debug:
                    raise
                self.response_queue.put({"command": "error", "result": repr(e), "elapsed_time": 0.0})
