"""
═══════════════════════════════════════════════════════════════════════════════
⚠️  DEPRECATED - 已弃用
═══════════════════════════════════════════════════════════════════════════════

Project Trinity - Mouth Adapter (CosyVoice 3.0)
嘴巴适配器 - 语音合成

【弃用说明】
- 此文件是 CosyVoice 3.0 专用的旧版适配器
- 项目已迁移到 MOSS-Speech + TensorRT-LLM 方案
- 新的语音合成请使用 server/trtllm_moss/ 目录下的实现
- 保留此文件仅供参考，请勿在新代码中使用

【弃用时间】2026-01-18

功能 (已弃用):
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
    """
    
    EMOTION_INSTRUCTIONS = {
        "Soft": "用温柔轻柔的声音说",
        "Concerned": "用关切担忧的语气说",
        "Playful": "用俏皮活泼的语调说",
        "Serious": "用认真严肃的声音说",
        "Flirty": "用撒娇甜蜜的语气说",
        "Defensive": "用有些紧张防御的声音说",
        "Neutral": "用自然的声音说"
    }
    
    def __init__(
        self, 
        model_path: str = "FunAudioLLM/CosyVoice2-0.5B",
        remote_url: Optional[str] = None
    ):
        super().__init__("MouthAdapter")
        self.model_path = model_path
        self.remote_url = remote_url
        self.remote_mode = False
        self.model = None
        self.default_speaker = None
        self._lock = asyncio.Lock()
        
    async def initialize(self) -> bool:
        """初始化 CosyVoice 模型"""
        if self.model_path == "REMOTE":
            self.remote_mode = True
            logger.info(f"MouthAdapter 运行在远程模式: {self.remote_url}")
            self.is_initialized = True
            return True

        try:
            logger.info(f"正在初始化 CosyVoice 模型: {self.model_path}")
            
            # 使用 CosyVoice3
            from cosyvoice.cli.cosyvoice import CosyVoice3
            self.model = CosyVoice3(self.model_path, load_trt=False)
            
                    self.is_initialized = True
            logger.success("CosyVoice 模型初始化成功")
                    return True
            
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
        """合成语音"""
        if not self.is_initialized:
            raise RuntimeError("MouthAdapter 未初始化")
        
        emotion_key = emotion_tag.strip("[]")
        instruction = self.EMOTION_INSTRUCTIONS.get(emotion_key, self.EMOTION_INSTRUCTIONS["Neutral"])
        
        # 远程模式
        if self.remote_mode:
            import aiohttp
        try:
                async with aiohttp.ClientSession() as session:
                    payload = {
                        "text": text,
                        "instruct_text": instruction  # 改为 instruct_text
                    }
                    # 注意: remote_url 已经是 http://xxx:9000/mouth
                    async with session.post(f"{self.remote_url}/tts", json=payload) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            audio_list = data.get("audio", [])
                            sr = data.get("sample_rate", 24000)
                            
                            audio_data = np.array(audio_list, dtype=np.float32)
                            duration = len(audio_data) / sr
                        
                        return SpeechResult(
                                audio_data=audio_data,
                                sample_rate=sr,
                                duration=duration
                        )
                else:
                            logger.error(f"Remote TTS Error: {resp.status}")
                            return self._mock_result()
            except Exception as e:
                logger.error(f"Remote TTS Exception: {e}")
                return self._mock_result()

        # 本地模式
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
                return self._mock_result()
    
    def _mock_result(self):
            return SpeechResult(
                audio_data=np.zeros(16000, dtype=np.float32),
                sample_rate=16000,
                duration=1.0
            )

    def _process_action_tags(self, text: str) -> str:
        action_mapping = {
            "[laugh]": "<laughter>",
            "[sigh]": "<sigh>",
            "[smile]": "",
            "[pause]": "...",
        }
        for tag, replacement in action_mapping.items():
            text = text.replace(tag, replacement)
        return text
    
    def _inference(self, text: str, instruction: str) -> SpeechResult:
        """同步推理"""
        # CosyVoice3 Inference
        prompt_wav = np.zeros(16000, dtype=np.float32)
        
        full_audio = []
        sample_rate = 24000
        
        for output in self.model.inference_instruct2(
            tts_text=text,
            instruct_text=instruction,
            prompt_wav=prompt_wav,
            stream=False
        ):
            if 'tts_speech' in output:
                full_audio.append(output['tts_speech'].cpu().numpy())
        
        if not full_audio:
            raise RuntimeError("No audio generated")
            
        audio_data = np.concatenate(full_audio, axis=1).flatten()
        duration = len(audio_data) / sample_rate
        
        return SpeechResult(
            audio_data=audio_data,
            sample_rate=sample_rate,
            duration=duration
        )

    async def shutdown(self) -> None:
        if self.model is not None and not isinstance(self.model, MockTTS):
            del self.model
            self.model = None
        self.is_initialized = False
        logger.info("MouthAdapter 已关闭")


class MockTTS:
    def inference_instruct2(self, *args, **kwargs):
        pass
