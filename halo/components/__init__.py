"""
Components 模块：Halo 系统的核心数据结构

该模块导出了 Halo 系统的核心数据结构类：
- Operator: 工作流图中的操作节点
- ModelConfig: 模型配置参数
- Query: 用户查询请求
- ExecuteInfo: 执行任务信息封装

这些组件构成了整个系统的基础数据模型。
"""

from .operator import Operator
from .model_config import ModelConfig
from .query import Query
from .exe_info import ExecuteInfo

# 导出所有公共类
__all__ = ["Operator", "ModelConfig", "Query", "ExecuteInfo"]