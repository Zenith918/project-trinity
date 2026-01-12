"""
Project Trinity - Voice Adapter (FunASR / SenseVoice)
听觉适配器 - Layer 1 的感知入口

功能:
- 实时语音识别 (ASR)
- 情感识别 (SER) - SenseVoice 原生支持
- 延迟 < 200ms
"""

from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import asyncio
import numpy as np
from loguru import logger

from .base_adapter import BaseAdapter


@dataclass
class VoiceResult:
    """语音识别结果"""
    text: str                          # 识别文本
    emotion: str                       # 情感标签 (happy/sad/angry/neutral/fearful)
    emotion_confidence: float          # 情感置信度
    language: str                      # 语言
    timestamps: Optional[list] = None  # 时间戳


class VoiceAdapter(BaseAdapter):
    """
    FunASR (SenseVoice) 适配器
    
    特性:
    - 支持流式识别
    - 原生情感识别 (SER)
    - 多语言支持
    - 支持远程模式 (连接到 Cortex-Ear)
    """
    
    def __init__(self, model_name: str = "iic/SenseVoiceSmall", device: str = "cuda:0", remote_url: Optional[str] = None):
        super().__init__("VoiceAdapter")
        self.model_name = model_name
        self.device = device
        self.model = None
        self.remote_url = remote_url
        self.remote_mode = False
        
    async def initialize(self) -> bool:
        """初始化 FunASR 模型或连接远程服务"""
        
        # 远程模式
        if self.model_name == "REMOTE":
            self.remote_mode = True
            logger.info(f"VoiceAdapter 运行在远程模式: {self.remote_url}")
            self.is_initialized = True
            return True
        
        # 本地模式
        try:
            logger.info(f"正在初始化 FunASR 模型: {self.model_name}")
            
            # 动态导入，避免启动时依赖问题
            from funasr import AutoModel
            
            self.model = AutoModel(
                model=self.model_name,
                trust_remote_code=True,
                device=self.device
            )
            
            self.is_initialized = True
            logger.success(f"FunASR 模型初始化成功: {self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"FunASR 初始化失败: {e}")
            return False
    
    async def process(self, audio_data: np.ndarray, sample_rate: int = 16000) -> VoiceResult:
        """
        处理音频数据
        
        Args:
            audio_data: 音频波形数据 (numpy array)
            sample_rate: 采样率
            
        Returns:
            VoiceResult: 包含文本和情感的识别结果
        """
        if not self.is_initialized:
            raise RuntimeError("VoiceAdapter 未初始化")
        
        # 远程模式: 调用 Cortex-Ear 服务
        if self.remote_mode:
            return await self._process_remote(audio_data, sample_rate)
        
        # 本地模式
        async with self._lock:
            try:
                # 在线程池中运行同步推理
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    self._inference,
                    audio_data,
                    sample_rate
                )
                return result
                
            except Exception as e:
                logger.error(f"语音处理失败: {e}")
                return VoiceResult(
                    text="",
                    emotion="neutral",
                    emotion_confidence=0.0,
                    language="unknown"
                )
    
    async def _process_remote(self, audio_data: np.ndarray, sample_rate: int) -> VoiceResult:
        """通过远程 Cortex-Ear 服务处理"""
        import aiohttp
        import io
        import wave
        
        try:
            # 将 numpy 数组转换为 WAV 字节
            audio_int16 = (audio_data * 32767).astype(np.int16) if audio_data.dtype == np.float32 else audio_data
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_int16.tobytes())
            wav_buffer.seek(0)
            
            # 发送到远程服务
            async with aiohttp.ClientSession() as session:
                data = aiohttp.FormData()
                data.add_field('file', wav_buffer, filename='audio.wav', content_type='audio/wav')
                
                async with session.post(f"{self.remote_url}/transcribe", data=data) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return VoiceResult(
                            text=result.get("text", ""),
                            emotion=result.get("emotion", "neutral"),
                            emotion_confidence=0.8,
                            language=result.get("language", "zh")
                        )
                    else:
                        logger.error(f"Remote ASR Error: {resp.status}")
                        return VoiceResult(text="", emotion="neutral", emotion_confidence=0.0, language="unknown")
                        
        except Exception as e:
            logger.error(f"Remote ASR 调用失败: {e}")
            return VoiceResult(text="", emotion="neutral", emotion_confidence=0.0, language="unknown")
    
    def _inference(self, audio_data: np.ndarray, sample_rate: int) -> VoiceResult:
        """同步推理（在线程池中执行）"""
        result = self.model.generate(
            input=audio_data,
            cache={},
            language="auto",
            use_itn=True,
            batch_size_s=60
        )
        
        # 解析 SenseVoice 的输出格式
        # SenseVoice 返回格式: "<|emotion|>text<|/emotion|>"
        text = result[0]["text"] if result else ""
        emotion, clean_text = self._parse_emotion(text)
        
        return VoiceResult(
            text=clean_text,
            emotion=emotion,
            emotion_confidence=0.8,  # SenseVoice 不返回置信度，使用默认值
            language="zh" if any('\u4e00' <= c <= '\u9fff' for c in clean_text) else "en"
        )
    
    def _parse_emotion(self, text: str) -> Tuple[str, str]:
        """
        解析 SenseVoice 的情感标签
        
        SenseVoice 输出格式: "<|HAPPY|>我今天很开心"
        """
        emotion_map = {
            "HAPPY": "happy",
            "SAD": "sad", 
            "ANGRY": "angry",
            "FEARFUL": "fearful",
            "DISGUSTED": "disgusted",
            "SURPRISED": "surprised",
            "NEUTRAL": "neutral"
        }
        
        emotion = "neutral"
        clean_text = text
        
        for tag, emotion_name in emotion_map.items():
            if f"<|{tag}|>" in text:
                emotion = emotion_name
                clean_text = text.replace(f"<|{tag}|>", "").strip()
                break
                
        return emotion, clean_text
    
    async def process_stream(self, audio_chunk: bytes):
        """
        流式处理音频（用于实时对话）
        
        TODO: Phase 1 实现流式 ASR
        """
        pass
    
    async def shutdown(self) -> None:
        """关闭模型"""
        if self.model is not None:
            del self.model
            self.model = None
        self.is_initialized = False
        logger.info("VoiceAdapter 已关闭")

