import uuid
class Operator:  
    def __init__(self, id=None, prompt=None, model_config=None, keep_cache=False):
        # 生成或使用提供的唯一标识符
        self.id = id if id is not None else uuid.uuid4()
        # 依赖关系：前驱节点和后继节点列表
        # 这些列表在 parser 构建图时会被填充，形成有向无环图（DAG）
        self.input_ops = []      # 前驱节点列表：该操作依赖哪些操作的输出
        self.output_ops = []    # 后继节点列表：该操作的输出会被哪些操作使用
        # 提示词和模型配置
        self.prompt = prompt                    # 可选的提示词，用于指导模型生成
        self.model_config = model_config         # 模型配置对象，包含模型名称、采样参数等
        self.benchmark = Benchmark()
        
        # 数据并行相关字段（由调度器在运行时设置）
        self.data_parallel = False              # 是否采用数据并行（多设备复制执行）
        self.is_duplicate = False                # 是否为复制节点（用于数据并行）
        self.main_op = None                     # 如果为复制节点，指向原始主节点
        self.duplicate_info = None              # 复制信息 [rep_idx, rep_total]
                                                # rep_idx: 当前副本在总副本中的索引
                                                # rep_total: 总副本数


class Benchmark:
    
    def __init__(self):
        self.init_time = 0      # 初始化时间：模型加载、参数设置等
        self.prefill_time = 0   # Prefill 阶段时间：处理输入 prompt 的时间
        self.generate_time = 0  # Generate 阶段时间：生成新 token 的时间
        
    def total_time(self):
        return self.init_time + self.prefill_time + self.generate_time
    
    def update(self, dict):
        # 累加各项时间指标，用于统计多次执行的总时间
        self.init_time += dict.get('init_time', 0.0)
        self.prefill_time += dict.get('prefill_time', 0.0)
        self.generate_time += dict.get('generate_time', 0.0)

    def __str__(self):
        return f"Init time: {self.init_time}, Prefill time: {self.prefill_time}, Generate time: {self.generate_time}, Total time: {self.total_time()}"