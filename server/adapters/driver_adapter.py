import asyncio
import json
import subprocess
import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import numpy as np
from loguru import logger
from .base_adapter import BaseAdapter

@dataclass 
class FLAMEParams:
    """FLAME 面部参数"""
    expression: np.ndarray   # 表情参数 (50维)
    jaw_pose: np.ndarray     # 下巴姿态 (3维)
    eye_pose: np.ndarray     # 眼球姿态 (6维)
    head_pose: np.ndarray    # 头部姿态 (6维)
    timestamp: float         # 时间戳

@dataclass
class MotionResult:
    """动作生成结果"""
    flame_sequence: List[FLAMEParams]  # FLAME 参数序列
    fps: int                            # 帧率
    duration: float                     # 时长

class DriverAdapter(BaseAdapter):
    """
    Project Trinity 统一驱动适配器
    
    功能:
    1. Audio2Motion: 使用 GeneFace++ 从音频生成 FLAME 参数 (实时)
    2. AssetGen: 使用 FastAvatar 从照片生成 3DGS 资产 (离线/初始化)
    """
    
    def __init__(self, geneface_path: str = "models/geneface"):
        super().__init__("DriverAdapter")
        self.geneface_path = geneface_path
        self.audio2motion_model = None
        self.fps = 25
        
        # FastAvatar 环境路径
        self.fastavatar_env_python = "/workspace/project-trinity/env_avatar/bin/python"
        self.fastavatar_script = "/workspace/project-trinity/server/adapters/fastavatar_runner.py"

    async def initialize(self) -> bool:
        """初始化驱动"""
        try:
            logger.info("正在初始化 DriverAdapter...")
            
            # 1. 初始化 GeneFace++ (Audio2Motion)
            # 实际集成时需 import geneface
            # from geneface.inference import Audio2Motion
            # self.audio2motion_model = Audio2Motion(self.geneface_path)
            
            logger.info("加载 GeneFace++ (Mock 模式)...")
            self.audio2motion_model = MockGeneFace()
            
            self.is_initialized = True
            logger.success("DriverAdapter 初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"DriverAdapter 初始化失败: {e}")
            return False

    async def process(self, audio_data: np.ndarray, sample_rate: int = 16000, base_emotion: str = "neutral") -> MotionResult:
        """处理音频生成动作"""
        if not self.is_initialized:
            raise RuntimeError("DriverAdapter 未初始化")
            
        # 调用 GeneFace++ 生成 FLAME
        return self.audio2motion_model.process(audio_data, sample_rate)

    async def generate_avatar(self, image_path: str, output_dir: str) -> bool:
        """
        调用隔离环境的 FastAvatar 生成 3DGS 资产
        """
        logger.info(f"正在生成 Avatar 资产: {image_path} -> {output_dir}")
        
        cmd = [
            self.fastavatar_env_python,
            "models/fastavatar_repo/scripts/inference_feedforward_no_guidance.py",
            "--image", image_path,
            "--output_dir", output_dir,
            "--encoder_checkpoint", "models/fastavatar/pretrained_weights/pretrained_weights/encoder_neutral_flame.pth",
            "--decoder_checkpoint", "models/fastavatar/pretrained_weights/pretrained_weights/decoder_neutral_flame.pth",
            "--dino_checkpoint", "models/fastavatar/pretrained_weights/pretrained_weights/dino_encoder.pth"
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.success(f"Avatar 生成成功: {output_dir}/splats.ply")
                return True
            else:
                logger.error(f"Avatar 生成失败:\n{stderr.decode()}")
                return False
        except Exception as e:
            logger.error(f"Avatar 生成进程错误: {e}")
            return False

    async def shutdown(self) -> None:
        """关闭驱动适配器"""
        if self.audio2motion_model:
            # 如果有 cleanup 方法则调用
            if hasattr(self.audio2motion_model, "cleanup"):
                self.audio2motion_model.cleanup()
            self.audio2motion_model = None
            
        self.is_initialized = False
        logger.info("DriverAdapter 已关闭")

class MockGeneFace:
    """模拟 GeneFace++"""
    def process(self, audio_data: np.ndarray, sample_rate: int) -> MotionResult:
        duration = len(audio_data) / sample_rate
        fps = 25
        num_frames = int(duration * fps)
        
        sequence = []
        for i in range(num_frames):
            # 简单的正弦波嘴部运动
            t = i / fps
            mouth_open = np.abs(np.sin(t * 10)) * 0.5 * (np.mean(np.abs(audio_data)) * 10)
            
            # 构造 FLAME 参数 (示意)
            expr = np.zeros(50)
            expr[0] = mouth_open # 假设第一个是张嘴
            
            sequence.append(FLAMEParams(
                expression=expr,
                jaw_pose=np.zeros(3),
                eye_pose=np.zeros(6),
                head_pose=np.zeros(6),
                timestamp=t
            ))
            
        return MotionResult(sequence, fps, duration)
