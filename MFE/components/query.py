import uuid
import time


class Query:

    def __init__(self, id, prompt, priority=0, template=""):
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
