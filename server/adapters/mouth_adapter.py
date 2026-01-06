"""
Project Trinity - Mouth Adapter (CosyVoice)
嘴巴适配器 - 语音合成

功能:
- 富情感语音合成
- 支持 [laugh], [sigh] 等指令
- 低延迟流式输出
"""

import sys
import os
from typing import Optional, Dict, Any, AsyncGenerator
from dataclasses import dataclass
import asyncio
import numpy as np
from loguru import logger

# 添加 CosyVoice 本地路径
COSYVOICE_PATH = "/workspace/CosyVoice"
if os.path.exists(COSYVOICE_PATH) and COSYVOICE_PATH not in sys.path:
    sys.path.insert(0, COSYVOICE_PATH)

from .base_adapter import BaseAdapter


@dataclass
class SpeechResult:
    """语音合成结果"""
    audio_data: np.ndarray  # 音频波形
    sample_rate: int        # 采样率
    duration: float         # 时长（秒）


class MouthAdapter(BaseAdapter):
    """
    CosyVoice 3.0 适配器
    
    特性:
    - Instruct Mode 支持情感指令
    - 零样本语音克隆
    - 流式输出
    """
    
    # 情感映射到 CosyVoice 指令
    EMOTION_INSTRUCTIONS = {
        "Soft": "用温柔轻柔的声音说",
        "Concerned": "用关切担忧的语气说",
        "Playful": "用俏皮活泼的语调说",
        "Serious": "用认真严肃的声音说",
        "Flirty": "用撒娇甜蜜的语气说",
        "Defensive": "用有些紧张防御的声音说",
        "Neutral": "用自然的声音说"
    }
    
    def __init__(self, model_path: str = "FunAudioLLM/CosyVoice2-0.5B"):
        super().__init__("MouthAdapter")
        self.model_path = model_path
        self.model = None
        self.default_speaker = None
        
    async def initialize(self) -> bool:
        """初始化 CosyVoice 模型"""
        try:
            logger.info(f"正在初始化 CosyVoice 模型: {self.model_path}")
            
            # 检查本地模型路径
            if os.path.exists(self.model_path):
                logger.info(f"使用本地模型: {self.model_path}")
            
            # CosyVoice 初始化 - 尝试多种导入方式
            cosyvoice_loaded = False
            
            # 方式1: CosyVoice CLI (推荐)
            try:
                from cosyvoice.cli.cosyvoice import CosyVoice
                self.model = CosyVoice(self.model_path)
                cosyvoice_loaded = True
                logger.info("使用 CosyVoice CLI 模式")
            except ImportError as e:
                logger.debug(f"CosyVoice CLI 导入失败: {e}")
            except Exception as e:
                logger.debug(f"CosyVoice CLI 初始化失败: {e}")
            
            # 方式2: CosyVoice2
            if not cosyvoice_loaded:
                try:
                    from cosyvoice.cli.cosyvoice import CosyVoice2
                    self.model = CosyVoice2(self.model_path, load_jit=False, load_trt=False)
                    cosyvoice_loaded = True
                    logger.info("使用 CosyVoice2 模式")
                except ImportError as e:
                    logger.debug(f"CosyVoice2 导入失败: {e}")
                except Exception as e:
                    logger.debug(f"CosyVoice2 初始化失败: {e}")
            
            if cosyvoice_loaded:
                self.is_initialized = True
                logger.success("CosyVoice 模型初始化成功")
                return True
            else:
                raise ImportError("无法导入 CosyVoice 模块")
            
        except Exception as e:
            logger.error(f"CosyVoice 初始化失败: {e}")
            # 使用 Mock 模式作为后备
            logger.warning("将使用 Mock TTS 模式")
            self.model = MockTTS()
            self.is_initialized = True
            return True
    
    async def process(
        self,
        text: str,
        emotion_tag: str = "Neutral",
        speaker_embedding: Optional[np.ndarray] = None
    ) -> SpeechResult:
        """
        合成语音
        
        Args:
            text: 要合成的文本
            emotion_tag: 情感标签
            speaker_embedding: 说话人嵌入（用于声音克隆）
            
        Returns:
            SpeechResult: 音频数据
        """
        if not self.is_initialized:
            raise RuntimeError("MouthAdapter 未初始化")
        
        # 获取情感指令
        emotion_key = emotion_tag.strip("[]")
        instruction = self.EMOTION_INSTRUCTIONS.get(emotion_key, self.EMOTION_INSTRUCTIONS["Neutral"])
        
        # 处理内嵌动作标签
        text = self._process_action_tags(text)
        
        async with self._lock:
            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    self._inference,
                    text,
                    instruction
                )
                return result
                
            except Exception as e:
                logger.error(f"语音合成失败: {e}")
                # 返回静音
                return SpeechResult(
                    audio_data=np.zeros(16000, dtype=np.float32),
                    sample_rate=16000,
                    duration=1.0
                )
    
    def _process_action_tags(self, text: str) -> str:
        """
        处理动作标签，转换为 CosyVoice 支持的格式
        
        例如: [laugh] -> <laugh>
        """
        action_mapping = {
            "[laugh]": "<laughter>",
            "[sigh]": "<sigh>",
            "[smile]": "",  # 微笑不影响语音
            "[pause]": "...",
        }
        
        for tag, replacement in action_mapping.items():
            text = text.replace(tag, replacement)
            
        return text
    
    def _inference(self, text: str, instruction: str) -> SpeechResult:
        """同步推理"""
        # CosyVoice Instruct Mode
        result = self.model.inference_instruct(
            text,
            instruction,
            self.default_speaker
        )
        
        audio_data = result["audio"]
        sample_rate = result.get("sample_rate", 22050)
        duration = len(audio_data) / sample_rate
        
        return SpeechResult(
            audio_data=audio_data,
            sample_rate=sample_rate,
            duration=duration
        )
    
    async def process_stream(
        self,
        text: str,
        emotion_tag: str = "Neutral"
    ) -> AsyncGenerator[bytes, None]:
        """
        流式语音合成
        
        用于实时对话场景，边合成边播放
        
        TODO: Phase 2 实现流式 TTS
        """
        pass
    
    def set_speaker(self, speaker_audio: np.ndarray, sample_rate: int) -> None:
        """
        设置说话人（用于声音克隆）
        
        Args:
            speaker_audio: 参考音频
            sample_rate: 采样率
        """
        # TODO: 实现声音克隆
        pass
    
    async def shutdown(self) -> None:
        """关闭模型"""
        if self.model is not None:
            del self.model
            self.model = None
        self.is_initialized = False
        logger.info("MouthAdapter 已关闭")


class MockTTS:
    """Mock TTS 实现（当 CosyVoice 不可用时）"""
    
    def inference_instruct(self, text: str, instruction: str, speaker) -> dict:
        """生成静音音频作为占位"""
        import numpy as np
        # 生成 1 秒静音
        sample_rate = 22050
        duration = max(1.0, len(text) * 0.1)  # 根据文本长度估算时长
        audio = np.zeros(int(sample_rate * duration), dtype=np.float32)
        
        return {
            "audio": audio,
            "sample_rate": sample_rate
        }

