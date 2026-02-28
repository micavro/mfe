"""
Operator 模块：定义工作流图中的操作节点（Operator）和性能基准（Benchmark）类

该模块是 Halo 系统的核心数据结构之一，用于表示智能体工作流中的每个操作节点。
每个 Operator 对应一个模型推理任务，包含模型配置、输入输出依赖关系等信息。
"""

import uuid


class Operator:
    """
    工作流图中的操作节点（Operator）类
    
    每个 Operator 表示智能体工作流中的一个节点，对应一个模型推理任务。
    节点之间通过 input_ops 和 output_ops 建立依赖关系，形成有向无环图（DAG）。
    
    Attributes:
        id: 操作符的唯一标识符，如果未提供则自动生成 UUID
        input_ops: 前驱节点列表，表示该操作依赖哪些操作的输出
        output_ops: 后继节点列表，表示该操作的输出会被哪些操作使用
        prompt: 可选的提示词字符串，用于指导模型生成
        model_config: ModelConfig 对象，包含模型名称、采样参数等配置信息
        max_distance: 到任意终点节点的最长路径长度（边数），由 parser 计算，用于调度优化
        keep_cache: 是否保留 KV cache，如果为 True，下游节点可以复用该节点的 cache
        benchmark: Benchmark 对象，记录该操作的性能指标（初始化时间、prefill 时间、生成时间）
        data_parallel: 是否采用数据并行（多设备复制执行），由调度器设置
        is_duplicate: 是否为复制节点（用于数据并行），True 表示这是原始节点的副本
        main_op: 如果为复制节点，指向原始主节点
        duplicate_info: 复制信息 [rep_idx, rep_total]，表示在总副本中的索引和总数
    """
    
    def __init__(self, id=None, prompt=None, model_config=None, keep_cache=False):
        """
        初始化一个 Operator 对象
        
        Args:
            id: 操作符的唯一标识符，如果为 None 则自动生成 UUID
            prompt: 可选的提示词字符串，用于指导模型生成
            model_config: ModelConfig 对象，包含模型相关配置（模型名称、采样参数等）
            keep_cache: 布尔值，指示是否保留 KV cache 供下游节点使用
                       如果为 True，下游节点可以复用该节点的 KV cache，避免重复计算前缀
        """
        # 生成或使用提供的唯一标识符
        self.id = id if id is not None else uuid.uuid4()
        
        # 依赖关系：前驱节点和后继节点列表
        # 这些列表在 parser 构建图时会被填充，形成有向无环图（DAG）
        self.input_ops = []      # 前驱节点列表：该操作依赖哪些操作的输出
        self.output_ops = []    # 后继节点列表：该操作的输出会被哪些操作使用
        
        # 提示词和模型配置
        self.prompt = prompt                    # 可选的提示词，用于指导模型生成
        self.model_config = model_config         # 模型配置对象，包含模型名称、采样参数等
        
        # 调度相关字段
        self.max_distance = None                # 到任意终点节点的最长路径长度，由 parser 计算
                                                # 用于调度器优化：距离终点越远的节点通常应该优先执行
        
        # Cache 管理 (vllm无用字段)
        self.keep_cache = keep_cache            # 是否保留 KV cache
                                                # 如果为 True，下游节点可以复用该节点的 cache
        
        # 性能基准：记录该操作的执行时间统计
        self.benchmark = Benchmark()
        
        # 数据并行相关字段（由调度器在运行时设置）
        self.data_parallel = False              # 是否采用数据并行（多设备复制执行）
        self.is_duplicate = False                # 是否为复制节点（用于数据并行）
        self.main_op = None                     # 如果为复制节点，指向原始主节点
        self.duplicate_info = None              # 复制信息 [rep_idx, rep_total]
                                                # rep_idx: 当前副本在总副本中的索引
                                                # rep_total: 总副本数


class Benchmark:
    """
    性能基准类：记录操作执行的时间统计信息
    
    用于跟踪和统计每个 Operator 的执行性能，包括初始化时间、prefill 时间和生成时间。
    这些信息可以用于性能分析和调度优化。
    
    Attributes:
        init_time: 初始化时间（秒），包括模型加载、参数设置等
        prefill_time: Prefill 阶段时间（秒），处理输入 prompt 的时间
        generate_time: Generate 阶段时间（秒），生成新 token 的时间
    """
    
    def __init__(self):
        """
        初始化性能基准对象，所有时间指标初始化为 0
        """
        self.init_time = 0      # 初始化时间：模型加载、参数设置等
        self.prefill_time = 0   # Prefill 阶段时间：处理输入 prompt 的时间
        self.generate_time = 0  # Generate 阶段时间：生成新 token 的时间
        
    def total_time(self):
        """
        计算总执行时间
        
        Returns:
            float: 总执行时间（init_time + prefill_time + generate_time）
        """
        return self.init_time + self.prefill_time + self.generate_time
    
    def update(self, dict):
        """
        更新性能指标，累加各项时间
        
        Args:
            dict: 包含时间指标的字典，键为 'init_time'、'prefill_time'、'generate_time'
                 如果某个键不存在，则使用默认值 0.0
        """
        # 累加各项时间指标，用于统计多次执行的总时间
        self.init_time += dict.get('init_time', 0.0)
        self.prefill_time += dict.get('prefill_time', 0.0)
        self.generate_time += dict.get('generate_time', 0.0)

    def __str__(self):
        """
        返回性能基准的字符串表示
        
        Returns:
            str: 格式化的性能统计信息字符串
        """
        return f"Init time: {self.init_time}, Prefill time: {self.prefill_time}, Generate time: {self.generate_time}, Total time: {self.total_time()}"