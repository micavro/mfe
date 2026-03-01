"""vLLM Worker：绑定 GPU，按需加载/切换模型，执行 ExecuteInfo 并返回结果。"""

from __future__ import annotations

import os
import time
import logging
import warnings
from typing import Any, Dict, List, Optional

warnings.filterwarnings("ignore", category=FutureWarning, message=".*pynvml.*")
import torch

from mfe.components import Query, Operator, ModelConfig, ExecuteInfo
from mfe.config import is_verbose


class vLLMWorker:
    def __init__(self, id: int, physical_gpu_id: int, cmd_queue: "Any", result_queue: "Any") -> None:
        self.id = id
        self.device = f"cuda:{physical_gpu_id}"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)
        self.model_name: Optional[str] = None
        self.llm = None
        self.tokenizer = None
        self.cmd_queue = cmd_queue
        self.response_queue = result_queue
        self.enforce_eager = True
        logging.info("vLLMWorker[%s] initialized on device %s", id, self.device)

    def init_op(self, op: Operator) -> None:
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        self.op_name = op.id
        self.is_duplicate = getattr(op, "is_duplicate", False)
        cfg: ModelConfig = op.model_config
        self.use_chat_template = bool(getattr(cfg, "use_chat_template", True))
        self.sampling_params = SamplingParams(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
            min_tokens=getattr(cfg, "min_tokens", 0),
        )
        model_name = cfg.model_name
        if model_name != self.model_name:
            if self.llm is not None:
                try:
                    del self.llm
                except Exception:
                    pass
                self.llm = None

            if self.tokenizer is not None:
                try:
                    del self.tokenizer
                except Exception:
                    pass
                self.tokenizer = None
            torch.cuda.empty_cache()
            self.llm = LLM(
                model_name,
                dtype=str(getattr(cfg, "dtype", "bfloat16")),
                quantization=getattr(cfg, "quantization", None),
                max_model_len=getattr(cfg, "max_model_len", None),
                enforce_eager=self.enforce_eager,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            self.model_name = model_name
        self.prefix = (getattr(cfg, "system_prompt", None) or "").strip()
        self.common_message: str = (getattr(cfg, "common_message", "") or "").strip()

    @torch.inference_mode()
    def execute(self, exe_info: ExecuteInfo) -> Dict[str, Any]:
        """init_op → 构建 batch 输入 → vLLM.generate → 返回 item/op_name/benchmark。"""
        init_start = time.perf_counter()
        self.init_op(exe_info.op)

        queries: List[Query] = [
            Query(qid, prompt) for qid, prompt in zip(exe_info.query_ids, exe_info.prompts)
        ]
        init_time = time.perf_counter() - init_start
        prefill_start = time.perf_counter()
        messages_batch: List[Any] = []
        if self.use_chat_template:
            sys_text = "\n".join([x for x in [self.prefix, self.common_message] if x]).strip()
            for q in queries:
                messages_batch.append([{"role": "system", "content": sys_text}, {"role": "user", "content": q.prompt}])
            inputs: List[str] = self.tokenizer.apply_chat_template(messages_batch, tokenize=False, add_generation_prompt=False)
        else:
            inputs = []
            for q in queries:
                header = "\n".join([x for x in [self.prefix, self.common_message] if x]).strip()
                inputs.append(f"{header}\n\n{q.prompt}" if header else q.prompt)

        outputs = self.llm.generate(inputs, self.sampling_params)  # type: ignore[arg-type]
        results: List[Dict[str, Any]] = []
        for i, output in enumerate(outputs):
            gen_text = output.outputs[0].text if output.outputs else ""
            if isinstance(inputs[i], str):
                full_text = inputs[i] + gen_text
            else:
                full_text = "".join([m.get("content", "") for m in inputs[i]]) + gen_text
            results.append({"id": queries[i].id, "output": full_text, "benchmark": (prefill_start, time.perf_counter())})

        generate_time = time.perf_counter() - prefill_start
        if self.is_duplicate:
            benchmark = {"init_time": 0.0, "prefill_time": 0.0, "generate_time": 0.0}
        else:
            benchmark = {"init_time": init_time, "prefill_time": 0.0, "generate_time": generate_time}

        return {"item": results, "op_name": self.op_name, "benchmark": benchmark}

    def exit(self) -> str:
        try:
            if self.llm is not None:
                del self.llm
        except Exception:
            pass
        try:
            if self.tokenizer is not None:
                del self.tokenizer
        except Exception:
            pass
        self.llm = None
        self.tokenizer = None
        torch.cuda.empty_cache()
        logging.info("vLLMWorker[%s] exited.", self.id)
        return "Worker exited."

    def run(self, debug: bool = True) -> None:
        """循环：cmd_queue 取命令 → execute/exit → result 写 response_queue。"""
        while True:
            msg = self.cmd_queue.get()
            if isinstance(msg, tuple):
                command, params = msg
            elif isinstance(msg, dict):
                command = msg.get("command")
                params = msg.get("params", ())
            else:
                self.response_queue.put({"command": "error", "result": "Unsupported message format", "elapsed_time": 0.0})
                continue
            if command == "exit":
                self.response_queue.put({"command": "exit", "result": self.exit(), "elapsed_time": 0.0})
                break
            if is_verbose() and command == "execute" and params:
                exe = params[0]
                op_id = getattr(exe.op, "id", "?")
                qids = getattr(exe, "query_ids", [])
                prompts = getattr(exe, "prompts", [])
                p0 = (prompts[0][:50] + "...") if prompts and len(prompts[0]) > 50 else (prompts[0] if prompts else "")
                print(f"[Worker {self.id}] recv execute op={op_id} query_ids={qids} n_prompts={len(prompts)} prompt0={p0!r}")
            func = getattr(self, command, None)
            if not callable(func):
                self.response_queue.put({"command": "error", "result": f"Unknown command: {command}", "elapsed_time": 0.0})
                continue
            start = time.perf_counter()
            try:
                result = func(*params) if params is not None else func()  # type: ignore[misc]
                elapsed = time.perf_counter() - start
                self.response_queue.put({"command": command, "result": result, "elapsed_time": elapsed})
                if is_verbose() and command == "execute" and isinstance(result, dict):
                    print(f"[Worker {self.id}] sent result op_name={result.get('op_name', '?')} elapsed={elapsed:.3f}s")
            except Exception as e:
                if debug:
                    raise
                self.response_queue.put({"command": "error", "result": repr(e), "elapsed_time": 0.0})


if __name__ == '__main__':
    import queue
    from mfe.components import Query, Operator, ModelConfig, ExecuteInfo

    query = Query(0, "What is the capital of France?")
    model = "meta-llama/Llama-3.2-3B-Instruct"

    worker = vLLMWorker(
        id='1',
        device='cuda:1',
        cmd_queue=queue.Queue(),
        result_queue=queue.Queue(),
    )
    
    config = ModelConfig(
        model_name=model,
        system_prompt='You are a helpful assistant.',
    )

    op = Operator(
        id='op_0',
        model_config=config,
        keep_cache=False,
    )
    
    exe_info = ExecuteInfo(
        op=op,
        query_ids=[query.id],
        prompts=[query.prompt],
    )
    out = worker.execute(exe_info)
    print(out)