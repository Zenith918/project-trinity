"""
Project Trinity - Orchestrator
主编排器 - 协调整个数据流转管线

数据流:
1. [Client] 麦克风 -> Opus -> WebSocket
2. [Server] FunASR -> 文本 + 情感
3. [Server] Layer 1-2-3 处理
4. [Server] CosyVoice -> 音频
5. [Server] GeneFace++ -> FLAME 参数
6. [Server] Packager -> 打包对齐
7. [Client] 渲染 + 播放
"""

from typing import Optional, Dict, Any, AsyncGenerator
from dataclasses import dataclass
import asyncio
from loguru import logger

from adapters import VoiceAdapter, BrainAdapter, MouthAdapter, DriverAdapter
from mind_engine import EgoDirector
from pipeline.packager import StreamPackager, MediaPacket


@dataclass
class PipelineResult:
    """管线处理结果"""
    text_response: str
    audio_packets: list
    flame_packets: list
    total_duration: float


class Orchestrator:
    """
    主编排器
    
    负责协调整个实时对话管线
    """
    
    def __init__(
        self,
        voice: VoiceAdapter,
        brain: BrainAdapter,
        mouth: MouthAdapter,
        driver: DriverAdapter,
        ego: EgoDirector
    ):
        self.voice = voice
        self.brain = brain
        self.mouth = mouth
        self.driver = driver
        self.ego = ego
        
        self.packager = StreamPackager()
        
        # 性能统计
        self._latency_samples = []
    
    async def process_audio_stream(
        self,
        audio_chunk: bytes
    ) -> AsyncGenerator[MediaPacket, None]:
        """
        处理音频流（主入口）
        
        这是实时对话的核心函数
        
        Args:
            audio_chunk: 音频数据块
            
        Yields:
            MediaPacket: 打包好的媒体数据
        """
        import time
        start_time = time.time()
        
        # === Stage 1: ASR + 情感识别 ===
        voice_result = await self.voice.process(audio_chunk)
        
        if not voice_result.text.strip():
            # 没有识别到文字，可能是静音
            return
        
        logger.debug(f"ASR: '{voice_result.text}' | 情感: {voice_result.emotion}")
        
        # === Stage 2: 三层心智处理 ===
        decision = await self.ego.process(
            user_text=voice_result.text,
            detected_emotion=voice_result.emotion
        )
        
        # 如果触发了反射，先发送微表情
        if decision.triggered_reflex:
            reflex_motion = await self.driver.generate_micro_expression(
                decision.triggered_reflex
            )
            for packet in self.packager.package_motion_only(reflex_motion):
                yield packet
        
        # === Stage 3: TTS 语音合成 ===
        speech = await self.mouth.process(
            text=decision.response_text,
            emotion_tag=decision.emotion_tag
        )
        
        # === Stage 4: 面部动画生成 ===
        motion = await self.driver.process(
            audio_data=speech.audio_data,
            sample_rate=speech.sample_rate,
            base_emotion=decision.emotion_tag.strip("[]").lower()
        )
        
        # === Stage 5: 打包对齐 ===
        packets = self.packager.package(
            audio_data=speech.audio_data,
            sample_rate=speech.sample_rate,
            flame_sequence=motion.flame_sequence,
            fps=motion.fps
        )
        
        # 记录延迟
        latency = time.time() - start_time
        self._latency_samples.append(latency)
        logger.debug(f"管线延迟: {latency*1000:.0f}ms")
        
        for packet in packets:
            yield packet
    
    async def process_text(self, text: str, emotion: str = "neutral") -> PipelineResult:
        """
        处理文本输入（用于测试或降级场景）
        
        Args:
            text: 用户文本
            emotion: 情感标签
            
        Returns:
            PipelineResult: 完整的处理结果
        """
        # 三层处理
        decision = await self.ego.process(
            user_text=text,
            detected_emotion=emotion
        )
        
        # TTS
        speech = await self.mouth.process(
            text=decision.response_text,
            emotion_tag=decision.emotion_tag
        )
        
        # 动画
        motion = await self.driver.process(
            audio_data=speech.audio_data,
            sample_rate=speech.sample_rate
        )
        
        # 打包
        packets = self.packager.package(
            audio_data=speech.audio_data,
            sample_rate=speech.sample_rate,
            flame_sequence=motion.flame_sequence,
            fps=motion.fps
        )
        
        return PipelineResult(
            text_response=decision.response_text,
            audio_packets=[p for p in packets if p.audio_data is not None],
            flame_packets=[p for p in packets if p.flame_params is not None],
            total_duration=speech.duration
        )
    
    def get_average_latency(self) -> float:
        """获取平均延迟"""
        if not self._latency_samples:
            return 0.0
        return sum(self._latency_samples) / len(self._latency_samples)
    
    def reset_stats(self) -> None:
        """重置统计"""
        self._latency_samples.clear()

