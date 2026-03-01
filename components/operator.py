"""DAG 节点：Operator、Benchmark。"""

import uuid


class Benchmark:
    def __init__(self):
        self.init_time = 0
        self.prefill_time = 0
        self.generate_time = 0

    def total_time(self):
        return self.init_time + self.prefill_time + self.generate_time

    def update(self, d):
        self.init_time += d.get("init_time", 0.0)
        self.prefill_time += d.get("prefill_time", 0.0)
        self.generate_time += d.get("generate_time", 0.0)

    def __str__(self):
        return f"Init: {self.init_time}, Prefill: {self.prefill_time}, Generate: {self.generate_time}, Total: {self.total_time()}"


class Operator:
    def __init__(self, id=None, prompt=None, model_config=None, keep_cache=False):
        self.id = id if id is not None else uuid.uuid4()
        self.input_ops = []
        self.output_ops = []
        self.prompt = prompt
        self.model_config = model_config
        self.benchmark = Benchmark()
        self.data_parallel = False
        self.is_duplicate = False
        self.main_op = None
        self.duplicate_info = None
