"""
ModelConfig 模块：定义模型配置类

该模块定义了 ModelConfig 类，用于封装模型推理所需的所有配置参数。
包括模型名称、采样参数、批处理参数、量化配置等。
"""


class ModelConfig:
    """
    模型配置类：封装模型推理所需的所有配置参数
    
    该类包含模型加载、推理执行所需的所有配置信息，包括模型标识、采样参数、
    批处理参数、量化配置等。每个 Operator 都有一个 ModelConfig 对象。
    
    Attributes:
        model_name: 模型标识符，可以是 HuggingFace 模型 ID（如 "meta-llama/Llama-3.1-8B-Instruct"）
                   或本地模型路径
        system_prompt: 系统级提示词，通常用于设置模型角色或行为
        common_message: 通用消息（可选），会与 system_prompt 一起使用
        temperature: 采样温度，控制生成的随机性
                    0.0 表示贪婪解码（总是选择概率最高的 token）
                    值越大，生成的随机性越高
        top_p: Top-p（Nucleus）采样参数，控制采样范围
               只从累积概率达到 top_p 的 token 集合中采样
        max_tokens: 单次生成的最大 token 数
        max_batch_size: 一次推理可以处理的最大查询数量
                        如果查询数量超过此值，可能需要分批处理或数据并行
        dtype: 模型数据类型，支持 "bfloat16"、"float16"、"float32" 等
        use_chat_template: 是否使用 HuggingFace 的聊天模板格式化输入
                          如果为 True，系统会将 system_prompt 和用户 prompt 组合成标准聊天格式
        quantization: 量化配置（如 AWQ、GPTQ），用于减少显存占用
        lora_config: LoRA 适配器配置，用于模型微调
        max_model_len: 模型支持的最大序列长度
        min_tokens: 单次生成的最小 token 数
    """
    
    def __init__(
        self, 
        model_name, 
        system_prompt=None, 
        temperature=0.7, 
        top_p=0.9, 
        max_tokens=256,
        max_batch_size=8,
        dtype='bfloat16',
        use_chat_template=True,
        quantization=None,
        lora_config=None,
        max_model_len=None,
        min_tokens=0,
        ):
        """
        初始化模型配置对象
        
        Args:
            model_name: 模型标识符，必需参数
            system_prompt: 系统提示词，可选
            temperature: 采样温度，默认 0.7
            top_p: Top-p 采样参数，默认 0.9
            max_tokens: 最大生成 token 数，默认 256
            max_batch_size: 最大批处理大小，默认 8
            dtype: 数据类型，默认 'bfloat16'
            use_chat_template: 是否使用聊天模板，默认 True
            quantization: 量化配置，默认 None
            lora_config: LoRA 配置，默认 None
            max_model_len: 最大模型长度，默认 None
            min_tokens: 最小生成 token 数，默认 0
        """
        # 模型标识和提示词
        self.model_name = model_name          # 模型标识符（HuggingFace ID 或本地路径）
        self.system_prompt = system_prompt    # 系统级提示词，用于设置模型角色或行为
        self.common_message = ''             # 通用消息（可选）
        
        # 采样参数：控制生成的质量和随机性
        self.temperature = temperature        # 采样温度：0.0 为贪婪解码，值越大随机性越高
        self.top_p = top_p                    # Top-p 采样：只从累积概率达到 top_p 的 token 集合中采样
        
        # 生成长度控制
        self.max_tokens = max_tokens          # 最大生成 token 数
        self.min_tokens = min_tokens          # 最小生成 token 数
        
        # 批处理参数
        self.max_batch_size = max_batch_size  # 最大批处理大小
                                             # 如果查询数量超过此值，可能需要分批处理或数据并行
        
        # 模型加载参数
        self.dtype = dtype                    # 数据类型：'bfloat16'、'float16'、'float32' 等
        self.quantization = quantization      # 量化配置（AWQ、GPTQ 等），用于减少显存占用
        self.lora_config = lora_config        # LoRA 适配器配置，用于模型微调
        self.max_model_len = max_model_len    # 模型支持的最大序列长度
        
        # 输入格式化
        self.use_chat_template = use_chat_template  # 是否使用 HuggingFace 聊天模板
                                                    # 如果为 True，会将 system_prompt 和用户 prompt
                                                    # 组合成标准的聊天格式（如 Llama 的 chat template）