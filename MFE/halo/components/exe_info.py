"""
ExecuteInfo 模块：定义执行信息封装类

该模块定义了 ExecuteInfo 类，用于封装一次执行任务所需的所有信息。
这是一个简单的数据容器，便于在进程间传递执行任务信息。
"""

from .operator import Operator


class ExecuteInfo:
    """
    执行信息封装类：用于在优化器（主进程）和 Worker 进程之间传递执行任务信息
    
    该类封装了一次执行任务所需的所有信息，包括要执行的 Operator、查询 ID 列表
    和对应的提示词列表。优化器根据调度计划创建 ExecuteInfo 对象，通过进程间队列
    发送给 Worker，Worker 接收到后执行推理任务。
    
    Attributes:
        op: 要执行的 Operator 对象，包含模型配置和执行参数
        query_ids: 查询 ID 列表，表示这次执行要处理哪些查询
        prompts: 提示词列表，与 query_ids 一一对应
                 注意：prompts 中的每个 prompt 已经包含了父 OP 的输出
                 （由优化器在发送前拼接），实现了多步推理的依赖关系
    """
    
    def __init__(self, op: Operator, query_ids, prompts):
        """
        初始化执行信息对象
        
        Args:
            op: 要执行的 Operator 对象，包含模型配置和执行参数
            query_ids: 查询 ID 列表，表示这次执行要处理哪些查询
            prompts: 提示词列表，与 query_ids 一一对应，长度必须相同
                     每个 prompt 已经包含了父 OP 的输出（由优化器拼接）
        """
        self.op = op                    # 要执行的 Operator
        self.query_ids = query_ids      # 查询 ID 列表
        self.prompts = prompts          # 对应的提示词列表（已包含父 OP 的输出）