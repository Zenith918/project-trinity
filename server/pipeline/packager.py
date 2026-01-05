"""
Project Trinity - Stream Packager
流式打包器 - 音频与动画对齐

核心职责:
- 将音频切片（200ms）与对应的 FLAME 参数打包
- 确保音画严格同步
- 使用 Protobuf 序列化（高效传输）
"""

from typing import List, Optional
from dataclasses import dataclass, field
import numpy as np
from loguru import logger


@dataclass
class MediaPacket:
    """
    媒体数据包
    
    传输格式: WebSocket Binary (Protobuf)
    """
    timestamp: float              # 时间戳（秒）
    duration: float               # 时长（秒）
    audio_data: Optional[bytes]   # 音频数据 (Opus 编码)
    flame_params: Optional[List[float]]  # FLAME 参数 (扁平化)
    is_final: bool = False        # 是否为最后一个包


class StreamPackager:
    """
    流式打包器
    
    将连续的音频流和 FLAME 参数序列切分成
    对齐的数据包，便于实时传输
    """
    
    # 每个包的时长（毫秒）
    PACKET_DURATION_MS = 200
    
    def __init__(self, audio_sample_rate: int = 16000):
        self.audio_sample_rate = audio_sample_rate
        
        # 每个包的音频样本数
        self.samples_per_packet = int(
            self.audio_sample_rate * self.PACKET_DURATION_MS / 1000
        )
    
    def package(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        flame_sequence: List,
        fps: int
    ) -> List[MediaPacket]:
        """
        将音频和 FLAME 参数打包对齐
        
        Args:
            audio_data: 完整音频波形
            sample_rate: 音频采样率
            flame_sequence: FLAME 参数序列
            fps: FLAME 参数帧率
            
        Returns:
            List[MediaPacket]: 对齐的数据包列表
        """
        packets = []
        
        # 计算总时长
        audio_duration = len(audio_data) / sample_rate
        packet_duration = self.PACKET_DURATION_MS / 1000
        
        # 计算包数量
        num_packets = int(np.ceil(audio_duration / packet_duration))
        
        # FLAME 帧到包的映射
        frames_per_packet = int(fps * packet_duration)
        
        for i in range(num_packets):
            timestamp = i * packet_duration
            
            # 提取音频切片
            audio_start = int(i * sample_rate * packet_duration)
            audio_end = int((i + 1) * sample_rate * packet_duration)
            audio_slice = audio_data[audio_start:audio_end]
            
            # 编码音频 (简化版，实际应使用 Opus)
            audio_bytes = self._encode_audio(audio_slice)
            
            # 提取 FLAME 参数切片
            flame_start = i * frames_per_packet
            flame_end = (i + 1) * frames_per_packet
            flame_slice = flame_sequence[flame_start:flame_end] if flame_start < len(flame_sequence) else []
            
            # 扁平化 FLAME 参数
            flame_flat = self._flatten_flame(flame_slice)
            
            packet = MediaPacket(
                timestamp=timestamp,
                duration=packet_duration,
                audio_data=audio_bytes,
                flame_params=flame_flat,
                is_final=(i == num_packets - 1)
            )
            
            packets.append(packet)
        
        logger.debug(f"打包完成: {len(packets)} 个数据包, 总时长 {audio_duration:.2f}s")
        
        return packets
    
    def package_motion_only(self, motion_result) -> List[MediaPacket]:
        """
        仅打包动画（用于微表情/反射弧）
        
        Args:
            motion_result: 动画结果 (MotionResult)
            
        Returns:
            List[MediaPacket]: 仅包含 FLAME 参数的数据包
        """
        packets = []
        packet_duration = self.PACKET_DURATION_MS / 1000
        
        # 计算包数量
        num_packets = int(np.ceil(motion_result.duration / packet_duration))
        frames_per_packet = int(motion_result.fps * packet_duration)
        
        for i in range(num_packets):
            timestamp = i * packet_duration
            
            # 提取 FLAME 参数
            flame_start = i * frames_per_packet
            flame_end = (i + 1) * frames_per_packet
            flame_slice = motion_result.flame_sequence[flame_start:flame_end]
            
            packet = MediaPacket(
                timestamp=timestamp,
                duration=packet_duration,
                audio_data=None,  # 无音频
                flame_params=self._flatten_flame(flame_slice),
                is_final=(i == num_packets - 1)
            )
            
            packets.append(packet)
        
        return packets
    
    def _encode_audio(self, audio_slice: np.ndarray) -> bytes:
        """
        编码音频为传输格式
        
        TODO: 使用 Opus 编码实现更好的压缩
        """
        # 简化版: 直接转为 bytes
        # 实际应使用 opuslib 或类似库
        return audio_slice.astype(np.float32).tobytes()
    
    def _flatten_flame(self, flame_slice: List) -> List[float]:
        """
        将 FLAME 参数扁平化
        
        Args:
            flame_slice: FLAMEParams 对象列表
            
        Returns:
            List[float]: 扁平化的参数列表
        """
        if not flame_slice:
            return []
        
        result = []
        for frame in flame_slice:
            # 按顺序拼接所有参数
            if hasattr(frame, 'expression'):
                result.extend(frame.expression.tolist() if hasattr(frame.expression, 'tolist') else list(frame.expression))
            if hasattr(frame, 'jaw_pose'):
                result.extend(frame.jaw_pose.tolist() if hasattr(frame.jaw_pose, 'tolist') else list(frame.jaw_pose))
            if hasattr(frame, 'eye_pose'):
                result.extend(frame.eye_pose.tolist() if hasattr(frame.eye_pose, 'tolist') else list(frame.eye_pose))
            if hasattr(frame, 'head_pose'):
                result.extend(frame.head_pose.tolist() if hasattr(frame.head_pose, 'tolist') else list(frame.head_pose))
        
        return result
    
    @staticmethod
    def decode_audio(audio_bytes: bytes) -> np.ndarray:
        """解码音频"""
        return np.frombuffer(audio_bytes, dtype=np.float32)
    
    @staticmethod
    def unflatten_flame(flat_params: List[float], num_frames: int) -> List[dict]:
        """
        反扁平化 FLAME 参数
        
        Args:
            flat_params: 扁平化的参数
            num_frames: 帧数
            
        Returns:
            List[dict]: FLAME 参数字典列表
        """
        if not flat_params or num_frames == 0:
            return []
        
        # 每帧的参数数量
        # expression(50) + jaw(3) + eye(6) + head(6) = 65
        params_per_frame = 65
        
        frames = []
        for i in range(num_frames):
            start = i * params_per_frame
            end = start + params_per_frame
            
            if end > len(flat_params):
                break
            
            frame_params = flat_params[start:end]
            
            frames.append({
                "expression": frame_params[0:50],
                "jaw_pose": frame_params[50:53],
                "eye_pose": frame_params[53:59],
                "head_pose": frame_params[59:65]
            })
        
        return frames

