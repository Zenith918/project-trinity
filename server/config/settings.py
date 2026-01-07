"""
Project Trinity - Configuration Settings
配置管理模块
"""

from pydantic_settings import BaseSettings
from typing import Optional
import os


class ServerSettings(BaseSettings):
    """服务端配置"""
    
    # 服务器设置
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # 显存监控阈值 (MB)
    vram_warning_threshold: int = 22000  # 22GB
    vram_critical_threshold: int = 23500 # 23.5GB
    
    # WebSocket 设置
    ws_max_connections: int = 100
    ws_heartbeat_interval: int = 30
    
    class Config:
        env_prefix = "TRINITY_"


class ModelSettings(BaseSettings):
    """AI 模型配置 - 2026 SOTA 版本 (RTX 4090 24GB 优化)"""
    
    # Qwen 2.5-VL (Brain) - 使用 AWQ 量化版以节省显存
    qwen_model_path: str = "/workspace/models/Qwen2.5-VL-7B-Instruct-AWQ"
    qwen_tensor_parallel_size: int = 1
    qwen_max_model_len: int = 4096  # 限制 context 以节省显存
    qwen_quantization: Optional[str] = "awq"
    
    # 显存利用率: 24GB * 0.5 ≈ 12GB for Qwen
    qwen_gpu_memory_utilization: float = 0.5
    
    # FunASR (Ears) - SenseVoice 本地路径
    funasr_model: str = "/workspace/models/SenseVoiceSmall"
    funasr_device: str = "cuda:0"
    
    # CosyVoice 3.0 (Mouth) - 使用 Fun-CosyVoice3-0.5B-2512
    cosyvoice_model_path: str = "/workspace/models/CosyVoice3-0.5B"
    
    # GeneFace++ (Driver) - Audio2Motion
    geneface_model_path: str = "/workspace/code/GeneFacePlusPlus"
    
    class Config:
        env_prefix = "MODEL_"


class BioStateSettings(BaseSettings):
    """生物状态系统配置 (Layer 1: The Id)"""
    
    # Big Five 人格特质默认值
    default_neuroticism: float = 0.5      # 神经质 (情绪稳定性)
    default_extraversion: float = 0.6     # 外向性
    default_openness: float = 0.7         # 开放性
    default_agreeableness: float = 0.8    # 宜人性
    default_conscientiousness: float = 0.5  # 尽责性
    
    # 内稳态参数
    cortisol_baseline: float = 30.0       # 皮质醇基线
    dopamine_baseline: float = 50.0       # 多巴胺基线
    
    # 概率采样参数
    state_update_sigma: float = 10.0      # 高斯分布标准差
    sensitivity_multiplier: float = 1.5   # 敏感度放大系数
    
    # Temperature 映射 (压力 -> LLM Temperature)
    temp_min: float = 0.1
    temp_max: float = 0.9
    
    class Config:
        env_prefix = "BIO_"


class MemorySettings(BaseSettings):
    """记忆系统配置 (Layer 2: The Superego)"""
    
    # Qdrant 向量数据库
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "trinity_memories"
    
    # Mem0 配置
    mem0_api_key: Optional[str] = None
    
    # 记忆检索
    memory_search_limit: int = 5
    memory_similarity_threshold: float = 0.7
    
    class Config:
        env_prefix = "MEMORY_"


class Settings:
    """全局配置聚合"""
    
    def __init__(self):
        self.server = ServerSettings()
        self.model = ModelSettings()
        self.bio_state = BioStateSettings()
        self.memory = MemorySettings()


# 全局配置实例
settings = Settings()