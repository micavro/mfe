"""
Halo 系统主模块

Halo 是一个用于高效执行智能体工作流的系统，支持：
- 多请求批处理
- 请求之间的依赖关系
- 多 GPU 并行执行
- 智能调度优化

主要模块：
- components: 核心数据结构（Operator、Query、ModelConfig、ExecuteInfo）
- parser: YAML 配置解析和 DAG 构建
- schedulers: 多种调度策略
- optimizers: 多进程协调器和执行引擎
- workers: 模型推理执行层
"""

__version__ = "1.0.0"
