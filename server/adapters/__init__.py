"""
Project Trinity - AI Model Adapters
AI 模型适配器包

使用 Adapter Pattern 封装所有 AI 模型，便于未来替换
"""

from .base_adapter import BaseAdapter
from .voice_adapter import VoiceAdapter
from .brain_adapter import BrainAdapter
from .mouth_adapter import MouthAdapter
from .driver_adapter import DriverAdapter

__all__ = [
    "BaseAdapter",
    "VoiceAdapter", 
    "BrainAdapter",
    "MouthAdapter",
    "DriverAdapter"
]

