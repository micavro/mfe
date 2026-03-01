
from .operator import Operator


class ExecuteInfo:
    
    def __init__(self, op: Operator, query_ids, prompts):
        self.op = op                    # 要执行的 Operator
        self.query_ids = query_ids      # 查询 ID 列表
        self.prompts = prompts          # 对应的提示词列表（已包含父 OP 的输出）