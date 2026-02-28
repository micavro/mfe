"""
Query 模块：定义用户查询请求类

该模块定义了 Query 类，用于表示一个用户查询请求。
Query 对象在整个工作流执行过程中维护查询的状态、中间结果和性能指标。
"""

import uuid
import time


class Query:
    """
    用户查询请求类

    表示一个用户查询请求，包含查询 ID、提示词、执行状态、中间结果等信息。
    在整个工作流执行过程中，Query 对象会跟踪该查询在每个 Operator 上的执行状态和输出。

    Attributes:
        id: 查询的唯一标识符，通常是整数或字符串，如果未提供则自动生成 UUID
        prompt: 用户输入的原始提示词字符串
        prompt_len: 提示词的长度（字符数），用于调度优化（短查询可能更快完成）
        status: 查询状态，可选值："pending"（待处理）、"running"（执行中）、"finished"（已完成）
        priority: 优先级，数值越大优先级越高，用于调度器排序
        template: 工作流 YAML 文件名或路径，用于 MFE 多工作流场景下选择该请求对应的 DAG

        工作流执行相关字段：
        op_output: 字典，键为 OP 的 ID（字符串），值为该 OP 对该查询产生的输出文本
                  当执行一个 OP 时，系统会从 op_output 中提取父 OP 的输出并拼接到当前 OP 的 prompt 后面
        step: 当前执行步骤（已完成的 OP 数量）

        性能基准：
        create_time: 查询创建的时间戳（用于计算总延迟）
        benchmark: 字典，键为 OP 的 ID，值为 (start_time, end_time) 元组，记录每个 OP 的执行时间范围
    """

    def __init__(self, id, prompt, priority=0, template=""):
        """
        初始化一个查询对象

        Args:
            id: 查询的唯一标识符（通常是整数或字符串），如果为 None 则自动生成 UUID
            prompt: 用户输入的原始提示词字符串
            priority: 优先级，默认为 0，数值越大优先级越高，调度器会按优先级降序排序
            template: 工作流 YAML 文件名（如 "adv_reason_3.yaml"）或路径，用于 MFE 多工作流场景
        """
        # 基本属性
        self.id = id if id is not None else uuid.uuid4()  # 唯一标识符
        self.prompt = prompt                                # 用户输入的提示词
        self.template = template or ""                      # 工作流模板名/路径（MFE）
        self.prompt_len = len(prompt) if prompt else 0       # 提示词长度，用于调度优化
        self.status = "pending"                             # 初始状态为待处理
        self.priority = priority                            # 优先级，数值越大优先级越高

        # 工作流执行相关字段
        self.op_output = {}   # 字典：op_id -> 该 OP 的输出文本，实现工作流依赖与多步推理
        self.step = 0         # 当前执行步骤（已完成的 OP 数量）

        # 性能基准
        self.create_time = time.perf_counter()  # 创建时间戳，用于计算总延迟
        self.benchmark = {}   # 字典：op_id -> (start_time, end_time)，记录每个 OP 的执行时间范围
