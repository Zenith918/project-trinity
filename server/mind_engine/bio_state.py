"""
Project Trinity - BioState (Layer 1: The Id)
本我层 - 概率内稳态与生物反射

这是"三位一体"架构的第一层，模拟生物的本能反应

核心概念:
1. 内稳态 (Homeostasis): Cortisol(压力), Dopamine(愉悦) 等生理指标
2. 概率采样: 不使用固定增量，而是基于高斯分布采样
3. 人格种子: Big Five 人格特质影响反应敏感度
4. RPE (Reward Prediction Error): 预期误差驱动状态更新

关键实现原则:
- 所有状态更新必须是概率性的
- 同样的刺激可能产生不同反应（这就是"生命感"）
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
import numpy as np
from loguru import logger


@dataclass
class PersonalitySeed:
    """
    Big Five 人格特质种子
    
    这些参数决定了 AI 的"脾气基调"
    """
    neuroticism: float = 0.5       # 神经质 (0-1): 高=情绪波动大
    extraversion: float = 0.6      # 外向性 (0-1): 高=主动表达
    openness: float = 0.7          # 开放性 (0-1): 高=想象力丰富
    agreeableness: float = 0.8     # 宜人性 (0-1): 高=温和体贴
    conscientiousness: float = 0.5 # 尽责性 (0-1): 高=认真负责


@dataclass
class BioStateSnapshot:
    """生物状态快照（用于传递给其他层）"""
    cortisol: float          # 皮质醇 (压力激素) 0-100
    dopamine: float          # 多巴胺 (愉悦) 0-100
    serotonin: float         # 血清素 (平静) 0-100
    adrenaline: float        # 肾上腺素 (警觉) 0-100
    current_mood: str        # 当前情绪标签
    temperature: float       # 推荐的 LLM Temperature


class BioState:
    """
    生物状态系统 (Layer 1: The Id)
    
    这是一个概率状态机，不是确定性的
    
    核心公式:
    1. Sensitivity = 1.0 + (Current_Cortisol * Neuroticism)
    2. RPE = |Actual_Emotion - Expected_Emotion|
    3. Delta = Sample(Normal_Dist) * Sensitivity * RPE
    """
    
    # 情绪到生理影响的映射
    EMOTION_EFFECTS = {
        "happy": {"cortisol": -10, "dopamine": 20, "serotonin": 10, "adrenaline": 0},
        "sad": {"cortisol": 15, "dopamine": -15, "serotonin": -10, "adrenaline": 0},
        "angry": {"cortisol": 25, "dopamine": -5, "serotonin": -20, "adrenaline": 30},
        "fearful": {"cortisol": 30, "dopamine": -10, "serotonin": -15, "adrenaline": 40},
        "surprised": {"cortisol": 10, "dopamine": 10, "serotonin": 0, "adrenaline": 20},
        "disgusted": {"cortisol": 15, "dopamine": -10, "serotonin": -5, "adrenaline": 10},
        "neutral": {"cortisol": 0, "dopamine": 0, "serotonin": 5, "adrenaline": -5},
    }
    
    def __init__(self, personality: Optional[PersonalitySeed] = None):
        """
        初始化生物状态系统
        
        Args:
            personality: 人格种子，决定反应的敏感度
        """
        self.personality = personality or PersonalitySeed()
        
        # 内稳态基线
        self._cortisol = 30.0    # 基线压力
        self._dopamine = 50.0    # 基线愉悦
        self._serotonin = 50.0   # 基线平静
        self._adrenaline = 20.0  # 基线警觉
        
        # 预期情绪（用于计算 RPE）
        self._expected_emotion = "neutral"
        
        # 随机数生成器（可设置种子以便调试）
        self._rng = np.random.default_rng()
        
        logger.info(f"BioState 初始化完成 | 人格: N={self.personality.neuroticism:.2f}")
    
    def update(self, detected_emotion: str, confidence: float = 0.8) -> Dict[str, float]:
        """
        根据检测到的情绪更新生物状态
        
        这是核心的概率更新函数
        
        Args:
            detected_emotion: 检测到的情绪 (来自 SenseVoice)
            confidence: 情绪检测置信度
            
        Returns:
            Dict: 状态变化量 {"cortisol_delta": ..., "triggered_reflex": ...}
        """
        # 1. 计算 RPE (预期误差)
        rpe = self._calculate_rpe(detected_emotion)
        
        # 2. 计算敏感度 (Sensitivity)
        sensitivity = self._calculate_sensitivity()
        
        # 3. 获取该情绪的基础影响
        base_effects = self.EMOTION_EFFECTS.get(detected_emotion, self.EMOTION_EFFECTS["neutral"])
        
        # 4. 概率采样更新
        deltas = {}
        for hormone, base_delta in base_effects.items():
            # 核心公式: Delta = Sample(Normal) * Sensitivity * RPE * Confidence
            sample = self._rng.normal(0, 10)  # 标准差为 10 的高斯采样
            actual_delta = (base_delta + sample) * sensitivity * rpe * confidence
            
            # 应用更新
            current = getattr(self, f"_{hormone}")
            new_value = np.clip(current + actual_delta, 0, 100)
            setattr(self, f"_{hormone}", new_value)
            
            deltas[f"{hormone}_delta"] = actual_delta
        
        # 5. 更新预期情绪
        self._expected_emotion = detected_emotion
        
        # 6. 检查是否触发反射
        reflex = self._check_reflex_trigger()
        deltas["triggered_reflex"] = reflex
        
        logger.debug(
            f"BioState 更新 | 情绪: {detected_emotion} | "
            f"RPE: {rpe:.2f} | 敏感度: {sensitivity:.2f} | "
            f"Cortisol: {self._cortisol:.1f} | Reflex: {reflex}"
        )
        
        return deltas
    
    def _calculate_rpe(self, actual_emotion: str) -> float:
        """
        计算预期误差 (Reward Prediction Error)
        
        预期和实际情绪差异越大，反应越强烈
        """
        # 情绪距离矩阵（简化版）
        emotion_distance = {
            ("neutral", "angry"): 0.8,
            ("neutral", "happy"): 0.5,
            ("happy", "sad"): 0.9,
            ("happy", "angry"): 0.7,
            # ... 其他组合
        }
        
        key = (self._expected_emotion, actual_emotion)
        reverse_key = (actual_emotion, self._expected_emotion)
        
        if key in emotion_distance:
            return emotion_distance[key]
        elif reverse_key in emotion_distance:
            return emotion_distance[reverse_key]
        elif self._expected_emotion == actual_emotion:
            return 0.3  # 预期匹配，低 RPE
        else:
            return 0.5  # 默认中等 RPE
    
    def _calculate_sensitivity(self) -> float:
        """
        计算当前敏感度
        
        公式: Sensitivity = 1.0 + (Current_Cortisol/100 * Neuroticism)
        
        高压力 + 高神经质 = 高敏感度 (容易情绪波动)
        """
        return 1.0 + (self._cortisol / 100) * self.personality.neuroticism
    
    def _check_reflex_trigger(self) -> Optional[str]:
        """
        检查是否触发反射弧
        
        这些反射不经过 LLM，直接触发微表情
        """
        # 压力突增 -> 防御性反应
        if self._cortisol > 70:
            return "defensive"  # 瞳孔收缩、后仰
        
        # 多巴胺飙升 -> 开心表情
        if self._dopamine > 80:
            return "joy"  # 眼睛发光、微笑
        
        # 肾上腺素飙升 -> 警觉
        if self._adrenaline > 60:
            return "alert"  # 眼睛睁大、身体前倾
        
        return None
    
    def get_snapshot(self) -> BioStateSnapshot:
        """获取当前状态快照"""
        return BioStateSnapshot(
            cortisol=self._cortisol,
            dopamine=self._dopamine,
            serotonin=self._serotonin,
            adrenaline=self._adrenaline,
            current_mood=self._determine_mood(),
            temperature=self._calculate_temperature()
        )
    
    def _determine_mood(self) -> str:
        """根据激素水平确定当前情绪"""
        if self._cortisol > 60:
            if self._adrenaline > 50:
                return "anxious"
            return "stressed"
        elif self._dopamine > 70:
            return "happy"
        elif self._serotonin > 60:
            return "calm"
        elif self._dopamine < 30:
            return "down"
        return "neutral"
    
    def _calculate_temperature(self) -> float:
        """
        计算推荐的 LLM Temperature
        
        公式: temperature = max(0.1, 1.0 - (cortisol / 100))
        
        高压力 = 低 Temperature (保守/防御)
        低压力 = 高 Temperature (创意/幽默)
        """
        return max(0.1, min(0.9, 1.0 - (self._cortisol / 100)))
    
    def decay(self, delta_time: float = 1.0) -> None:
        """
        自然衰减（趋向基线）
        
        Args:
            delta_time: 经过的时间（秒）
        """
        decay_rate = 0.02 * delta_time
        
        # 向基线衰减
        self._cortisol += (30.0 - self._cortisol) * decay_rate
        self._dopamine += (50.0 - self._dopamine) * decay_rate
        self._serotonin += (50.0 - self._serotonin) * decay_rate
        self._adrenaline += (20.0 - self._adrenaline) * decay_rate
    
    def inject_stimulus(self, stimulus_type: str, intensity: float = 1.0) -> None:
        """
        直接注入刺激（用于测试或特殊事件）
        
        Args:
            stimulus_type: 刺激类型 ("stress", "reward", "danger")
            intensity: 强度 (0-1)
        """
        if stimulus_type == "stress":
            self._cortisol = min(100, self._cortisol + 30 * intensity)
        elif stimulus_type == "reward":
            self._dopamine = min(100, self._dopamine + 30 * intensity)
        elif stimulus_type == "danger":
            self._adrenaline = min(100, self._adrenaline + 40 * intensity)
            self._cortisol = min(100, self._cortisol + 20 * intensity)
    
    def __repr__(self) -> str:
        return (
            f"BioState(cortisol={self._cortisol:.1f}, dopamine={self._dopamine:.1f}, "
            f"mood={self._determine_mood()}, temp={self._calculate_temperature():.2f})"
        )

