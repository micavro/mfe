"""单次执行任务：op、query_ids、prompts。"""

from .operator import Operator


class ExecuteInfo:
    def __init__(self, op: Operator, query_ids, prompts):
        self.op = op
        self.query_ids = query_ids
        self.prompts = prompts
