"""
vLLMWorker 模块：基于 vLLM 的 Worker 实现

该模块实现了 vLLMWorker 类，是一个独立的进程，绑定到特定的 GPU，
负责执行模型推理任务。Worker 通过队列与主进程（Optimizer）通信。

主要功能：
- 加载和管理模型（按需加载和切换）
- 接收执行任务（ExecuteInfo）
- 执行批量推理（使用 vLLM）
- 返回结果给主进程

vLLM 的优势：
- 连续批处理：自动管理动态批处理
- PagedAttention：高效的 KV cache 管理
- 高吞吐：针对批量推理优化
"""

from __future__ import annotations

import os
import time
import logging
import queue
import warnings
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", category=FutureWarning, message=".*pynvml.*")
import torch

# Project-local types
from mfe.components import Query, Operator, ModelConfig, ExecuteInfo
from mfe.config import is_verbose

# vLLM / transformers 延迟导入，避免仅使用 TestWorker 时因未安装 vLLM 而报错


class vLLMWorker:
    """
    基于 vLLM 的 Worker 进程
    
    Worker 是一个独立的进程，绑定到特定的 GPU，负责执行模型推理任务。
    通过队列与主进程（Optimizer）通信，接收任务并返回结果。
    
    主要职责：
    1. 初始化或切换模型/tokenizer（按每个 Operator 的需求）
    2. 构建输入提示词（可选使用 HuggingFace 聊天模板）
    3. 通过 vLLM 执行批量生成
    4. 以统一的字典格式返回结果给主进程
    
    消息格式：
    所有返回给主进程的消息必须是字典格式：
        {"command": <str>, "result": <payload>, "elapsed_time": <float>}
    
    Attributes:
        id: Worker ID（逻辑标识）
        device: 设备字符串（如 "cuda:0"）
        model_name: 当前加载的模型名称
        llm: vLLM 的 LLM 对象
        tokenizer: HuggingFace tokenizer 对象
        cmd_queue: 命令队列，接收来自 Optimizer 的任务
        response_queue: 结果队列，返回结果给 Optimizer
        enforce_eager: 是否禁用 CUDA 图优化（提高兼容性）
    """

    def __init__(
        self,
        id: int,
        physical_gpu_id: int,
        cmd_queue: "queue.Queue",
        result_queue: "queue.Queue",
    ) -> None:
        """
        初始化 Worker，绑定到指定 GPU
        
        Args:
            id: Worker ID（逻辑标识）
            physical_gpu_id: 物理 GPU ID（Worker 会绑定到此 GPU）
            cmd_queue: 命令队列，Optimizer 通过此队列发送任务
            result_queue: 结果队列，Worker 通过此队列返回结果
        """
        self.id = id
        self.device = f"cuda:{physical_gpu_id}"

        # 绑定到特定 GPU：通过设置环境变量，确保该进程只能看到指定的 GPU
        # 这是多 GPU 并行执行的关键：每个 Worker 进程只使用一个 GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)

        # 运行时状态：模型和 tokenizer 初始为 None，按需加载
        self.model_name: Optional[str] = None      # 当前加载的模型名称
        self.llm = None                            # vLLM 的 LLM 对象（延迟导入）
        self.tokenizer = None                     # HuggingFace tokenizer（延迟导入）

        # 进程间通信队列
        self.cmd_queue = cmd_queue          # 命令队列：接收任务
        self.response_queue = result_queue  # 结果队列：返回结果

        # vLLM 选项
        self.enforce_eager: bool = True     # 禁用 CUDA 图优化，提高兼容性

        logging.info("vLLMWorker[%s] initialized on device %s", id, self.device)

    # --------------------------------------------------------------------- #
    # Per-Operator initialization
    # --------------------------------------------------------------------- #

    def init_op(self, op: Operator) -> None:
        """
        为每个 OP 初始化运行时状态
        
        在执行每个 OP 之前调用，用于：
        1. 设置采样参数
        2. 设置系统提示词和通用消息
        3. 如果需要，切换模型/tokenizer（如果新 OP 使用不同的模型）
        
        Args:
            op: 要执行的 Operator 对象，包含模型配置和执行参数
            
        Note:
            模型切换逻辑：
            - 比较当前模型名称和新 OP 的模型名称
            - 如果不同，释放旧模型并加载新模型
            - 清空 CUDA 缓存以释放显存
            - 这确保了 Worker 可以处理使用不同模型的工作流
        """
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        self.op_name = op.id
        self.is_duplicate = getattr(op, "is_duplicate", False)  # 是否为复制节点

        cfg: ModelConfig = op.model_config
        # 是否使用 HuggingFace 聊天模板
        self.use_chat_template: bool = bool(getattr(cfg, "use_chat_template", True))

        # 构建 vLLM 采样参数：从 ModelConfig 中提取参数
        self.sampling_params = SamplingParams(
            temperature=cfg.temperature,    # 采样温度：控制生成的随机性
            top_p=cfg.top_p,                 # Top-p 采样：控制采样范围
            max_tokens=cfg.max_tokens,       # 最大生成 token 数
            min_tokens=getattr(cfg, "min_tokens", 0),  # 最小生成 token 数
        )

        # 模型切换逻辑：如果需要，加载或切换模型
        model_name = cfg.model_name
        if model_name != self.model_name:
            # 释放之前的模型实例，释放 GPU 显存
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

            # 清空 CUDA 缓存，确保显存被释放
            torch.cuda.empty_cache()

            # 加载新模型：vLLM 接受 dtype 作为字符串
            self.llm = LLM(
                model_name,
                dtype=str(getattr(cfg, "dtype", "bfloat16")),  # 数据类型
                quantization=getattr(cfg, "quantization", None),  # 量化配置
                max_model_len=getattr(cfg, "max_model_len", None),  # 最大模型长度
                enforce_eager=self.enforce_eager,  # 禁用 CUDA 图优化
            )
            # 加载 tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            self.model_name = model_name  # 更新当前模型名称

        # 系统提示词和通用消息（可选）
        self.prefix: str = (getattr(cfg, "system_prompt", None) or "").strip()
        self.common_message: str = (getattr(cfg, "common_message", "") or "").strip()

    # --------------------------------------------------------------------- #
    # Commands
    # --------------------------------------------------------------------- #

    @torch.inference_mode()
    def execute(self, exe_info: ExecuteInfo) -> Dict[str, Any]:
        """
        Execute a single round for one Operator:
          1) Initialize per-op state
          2) Build batch inputs (with or without chat template)
          3) Call vLLM.generate
          4) Return structured results

        Returns:
            {
              "item": List[{"id": int, "output": str, "benchmark": (float, float)}],
              "op_name": str,
              "benchmark": {"init_time": float, "prefill_time": float, "generate_time": float}
            }
        """
        init_start = time.perf_counter()
        self.init_op(exe_info.op)

        queries: List[Query] = [
            Query(qid, prompt) for qid, prompt in zip(exe_info.query_ids, exe_info.prompts)
        ]
        init_time = time.perf_counter() - init_start

        # 构建输入：根据是否使用聊天模板，构建不同格式的输入
        prefill_start = time.perf_counter()
        messages_batch: List[Any] = []
        if self.use_chat_template:
            # 使用 HuggingFace 聊天模板：将系统提示词和用户提示词组合成标准聊天格式
            # 例如 Llama 的格式：<|system|>...<|user|>...
            sys_text = "\n".join([x for x in [self.prefix, self.common_message] if x]).strip()
            for q in queries:
                messages_batch.append(
                    [
                        {"role": "system", "content": sys_text},  # 系统提示词
                        {"role": "user", "content": q.prompt},    # 用户提示词（已包含父 OP 的输出）
                    ]
                )
            # 应用聊天模板，转换为模型期望的格式
            inputs: List[str] = self.tokenizer.apply_chat_template(
                messages_batch,
                tokenize=False,              # 不进行 tokenization（vLLM 会处理）
                add_generation_prompt=False,  # 不添加生成提示符
            )
        else:
            # 普通字符串提示词：简单拼接系统提示词和用户提示词
            inputs: List[str] = []
            for q in queries:
                header = "\n".join([x for x in [self.prefix, self.common_message] if x]).strip()
                if header:
                    inputs.append(f"{header}\n\n{q.prompt}")  # 系统提示词 + 用户提示词
                else:
                    inputs.append(q.prompt)  # 只有用户提示词

        # 批量推理：调用 vLLM 的 generate 方法
        # vLLM 会自动处理批处理和 KV cache，返回生成结果列表
        outputs = self.llm.generate(inputs, self.sampling_params)  # type: ignore[arg-type]

        # 收集结果：提取每个查询的生成文本
        results: List[Dict[str, Any]] = []
        for i, output in enumerate(outputs):
            # 提取生成的文本
            gen_text = output.outputs[0].text if output.outputs else ""
            # 为了可重现性，将完整输入和生成文本拼接
            # 这样返回的结果包含了完整的上下文
            if isinstance(inputs[i], str):
                full_text = inputs[i] + gen_text
            else:
                # 回退处理（如果 apply_chat_template 返回的不是字符串）
                full_text = "".join([m.get("content", "") for m in inputs[i]]) + gen_text

            results.append(
                {
                    "id": queries[i].id,                    # 查询 ID
                    "output": full_text,                    # 完整输出（输入 + 生成）
                    "benchmark": (prefill_start, time.perf_counter()),  # 性能基准（时间戳）
                }
            )

        # 计算生成时间
        generate_time = time.perf_counter() - prefill_start
        # 构建基准数据：如果是复制节点，不记录基准（避免重复计算）
        if self.is_duplicate:
            benchmark = {"init_time": 0.0, "prefill_time": 0.0, "generate_time": 0.0}
        else:
            # 注意：vLLM 的 prefill 和 generate 是融合的，所以 prefill_time 为 0
            benchmark = {
                "init_time": init_time,      # 初始化时间（模型加载等）
                "prefill_time": 0.0,          # Prefill 时间（vLLM 中为 0）
                "generate_time": generate_time  # 生成时间
            }

        return {"item": results, "op_name": self.op_name, "benchmark": benchmark}

    def exit(self) -> str:
        """
        清理资源并退出 Worker
        
        释放模型和 tokenizer 对象，清空 CUDA 缓存，确保资源被正确释放。
        
        Returns:
            str: 退出确认消息
        """
        # 释放模型对象
        try:
            if self.llm is not None:
                del self.llm
        except Exception:
            pass
        # 释放 tokenizer 对象
        try:
            if self.tokenizer is not None:
                del self.tokenizer
        except Exception:
            pass
        # 清空引用
        self.llm = None
        self.tokenizer = None
        # 清空 CUDA 缓存，释放显存
        torch.cuda.empty_cache()
        logging.info("vLLMWorker[%s] exited.", self.id)
        return "Worker exited."

    # --------------------------------------------------------------------- #
    # Main loop
    # --------------------------------------------------------------------- #

    def run(self, debug: bool = True) -> None:
        """
        Worker 的主循环：持续监听命令队列并执行任务
        
        这是 Worker 进程的核心循环，会一直运行直到收到 "exit" 命令。
        循环中会：
        1. 从 cmd_queue 阻塞等待命令
        2. 解析命令格式
        3. 执行对应的命令（如 execute）
        4. 将结果通过 response_queue 返回给主进程
        
        支持的命令：
        - "execute": 执行推理任务
            params = (ExecuteInfo,)
            返回：包含结果和基准数据的字典
        - "exit": 退出 Worker
            params = ()
            返回：退出确认消息
        
        消息格式：
        所有消息都遵循统一格式：
            {"command": "<command_name>", "result": <result_payload>, "elapsed_time": <float>}
        
        Args:
            debug: 是否在调试模式下运行
                  如果为 True，遇到异常时会抛出；否则会返回错误消息
        """
        while True:
            # 阻塞等待命令（从 Optimizer 发送）
            msg = self.cmd_queue.get()
            
            # 解析消息格式：支持元组和字典两种格式
            if isinstance(msg, tuple):
                command, params = msg
            elif isinstance(msg, dict):
                command = msg.get("command")
                params = msg.get("params", ())
            else:
                # 不支持的消息格式，返回错误
                self.response_queue.put(
                    {"command": "error", "result": "Unsupported message format", "elapsed_time": 0.0}
                )
                continue

            # 处理退出命令
            if command == "exit":
                out = self.exit()  # 清理资源
                self.response_queue.put({"command": "exit", "result": out, "elapsed_time": 0.0})
                break  # 退出循环

            if is_verbose() and command == "execute" and params:
                exe = params[0]
                op_id = getattr(exe.op, "id", "?")
                qids = getattr(exe, "query_ids", [])
                prompts = getattr(exe, "prompts", [])
                p0 = (prompts[0][:50] + "...") if prompts and len(prompts[0]) > 50 else (prompts[0] if prompts else "")
                print(f"[Worker {self.id}] recv execute op={op_id} query_ids={qids} n_prompts={len(prompts)} prompt0={p0!r}")

            # 动态调用对应方法：通过 getattr 获取方法
            func = getattr(self, command, None)
            if not callable(func):
                # 未知命令，返回错误
                self.response_queue.put(
                    {"command": "error", "result": f"Unknown command: {command}", "elapsed_time": 0.0}
                )
                continue

            # 执行命令并返回结果
            start = time.perf_counter()
            try:
                # 调用对应方法（如 execute）
                result = func(*params) if params is not None else func()  # type: ignore[misc]
                elapsed = time.perf_counter() - start
                # 返回成功结果
                self.response_queue.put({"command": command, "result": result, "elapsed_time": elapsed})
                if is_verbose() and command == "execute" and isinstance(result, dict):
                    print(f"[Worker {self.id}] sent result op_name={result.get('op_name', '?')} elapsed={elapsed:.3f}s")
            except Exception as e:
                if debug:
                    # 调试模式：抛出异常，便于调试
                    raise
                # 非调试模式：返回错误消息
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