"""
Project Trinity - Base Adapter Interface
基础适配器接口

所有 AI 模型适配器的抽象基类，实现 Adapter Pattern
便于未来替换底层模型（如 Qwen -> GPT-6）
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
import asyncio
from loguru import logger


class BaseAdapter(ABC):
    """
    AI 模型适配器基类
    
    所有适配器必须实现:
    - initialize(): 初始化模型
    - process(): 处理输入
    - shutdown(): 关闭资源
    """
    
    def __init__(self, name: str):
        self.name = name
        self.is_initialized = False
        self._lock = asyncio.Lock()
        
    @abstractmethod
    async def initialize(self) -> bool:
        """
        初始化模型
        
        Returns:
            bool: 初始化是否成功
        """
        pass
    
    @abstractmethod
    async def process(self, input_data: Any) -> Any:
        """
        处理输入数据
        
        Args:
            input_data: 输入数据
            
        Returns:
            处理结果
        """
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """关闭并释放资源"""
        pass
    
    async def health_check(self) -> bool:
        """健康检查"""
        return self.is_initialized
    
    def __repr__(self) -> str:
        status = "initialized" if self.is_initialized else "not initialized"
        return f"<{self.__class__.__name__}({self.name}) [{status}]>"

