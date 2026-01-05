"""
Project Trinity - EgoDirector (Layer 3: The Ego)
自我层 - 决策与仲裁

这是"三位一体"架构的最高层，负责:
1. 调和"本我"的冲动和"超我"的约束
2. 在毫秒级 (System 1) 和秒级 (System 2) 做出反应
3. 生成最终的回复策略

核心职责:
- 接收 Layer 1 (BioState) 的生理状态
- 接收 Layer 2 (NarrativeManager) 的记忆和约束
- 调用 Brain (Qwen VL) 生成最终回复
"""

from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import asyncio
from loguru import logger

from ..adapters.brain_adapter import BrainAdapter, BrainResponse
from .bio_state import BioState, BioStateSnapshot
from .narrative_mgr import NarrativeManager


@dataclass
class DirectorDecision:
    """导演决策结果"""
    response_text: str          # 最终回复文本
    emotion_tag: str            # 情感标签
    action_hints: list          # 动作提示
    inner_monologue: str        # 内心独白（调试用）
    triggered_reflex: str       # 触发的反射（如果有）
    llm_temperature: float      # 使用的 Temperature


class EgoDirector:
    """
    自我导演 (Layer 3: The Ego)
    
    这是最终的决策者，协调三层架构
    
    工作流程:
    1. 接收用户输入
    2. 查询 Layer 1 获取生理状态 -> 决定 Temperature
    3. 查询 Layer 2 获取记忆上下文 -> 注入 Prompt
    4. 调用 Brain 生成回复
    5. 返回决策结果
    """
    
    def __init__(
        self,
        brain: BrainAdapter,
        bio_state: BioState,
        narrative_mgr: NarrativeManager
    ):
        self.brain = brain
        self.bio_state = bio_state
        self.narrative_mgr = narrative_mgr
        
        # 专家模型（MoA: Mixture of Agents）
        self.expert_adapter = None  # DeepSeek for hard tasks
        
    async def process(
        self,
        user_text: str,
        detected_emotion: str = "neutral",
        visual_context: Optional[str] = None
    ) -> DirectorDecision:
        """
        处理用户输入，生成完整的响应决策
        
        Args:
            user_text: 用户文本
            detected_emotion: 检测到的情绪（来自 SenseVoice）
            visual_context: 视觉场景描述（来自 Qwen VL）
            
        Returns:
            DirectorDecision: 完整的决策结果
        """
        # === Step 1: Layer 1 处理 - 更新生理状态 ===
        bio_deltas = self.bio_state.update(detected_emotion)
        triggered_reflex = bio_deltas.get("triggered_reflex")
        
        # 获取状态快照
        bio_snapshot = self.bio_state.get_snapshot()
        llm_temperature = bio_snapshot.temperature
        
        logger.debug(
            f"Layer 1 处理完成 | 情绪: {detected_emotion} | "
            f"Temperature: {llm_temperature:.2f} | Reflex: {triggered_reflex}"
        )
        
        # === Step 2: Layer 2 处理 - 获取记忆上下文 ===
        narrative_context = await self.narrative_mgr.get_context_for_response(
            user_text,
            bio_snapshot.current_mood
        )
        
        # 组装记忆上下文字符串
        memory_context = self._format_memory_context(narrative_context)
        
        logger.debug(f"Layer 2 处理完成 | 记忆数: {len(narrative_context['relevant_memories'])}")
        
        # === Step 3: 检查是否需要专家介入 (MoA) ===
        intent = await self._classify_intent(user_text)
        
        if intent == "HARD_TASK" and self.expert_adapter is not None:
            # 调用专家模型
            expert_result = await self._call_expert(user_text)
            # 将专家结果注入到 memory_context
            memory_context += f"\n[Expert Analysis]: {expert_result}"
        
        # === Step 4: Layer 3 处理 - 调用大脑生成回复 ===
        brain_response = await self.brain.process(
            user_input=user_text,
            visual_context=visual_context,
            bio_state={
                "cortisol": bio_snapshot.cortisol,
                "dopamine": bio_snapshot.dopamine,
                "mood": bio_snapshot.current_mood
            },
            memory_context=memory_context,
            temperature=llm_temperature
        )
        
        # === Step 5: 异步记录记忆（不阻塞响应）===
        asyncio.create_task(
            self._log_interaction(user_text, brain_response.response, detected_emotion)
        )
        
        return DirectorDecision(
            response_text=brain_response.response,
            emotion_tag=brain_response.emotion_tag,
            action_hints=brain_response.action_hints,
            inner_monologue=brain_response.inner_monologue,
            triggered_reflex=triggered_reflex,
            llm_temperature=llm_temperature
        )
    
    def _format_memory_context(self, narrative_context: Dict) -> str:
        """格式化记忆上下文"""
        parts = []
        
        # 相关记忆
        memories = narrative_context.get("relevant_memories", [])
        if memories:
            parts.append("Relevant memories:")
            for i, mem in enumerate(memories[:3]):
                parts.append(f"  - {mem}")
        
        # 人设约束
        constraints = narrative_context.get("persona_constraints", [])
        if constraints:
            parts.append("Persona constraints:")
            for c in constraints:
                parts.append(f"  - {c}")
        
        # 禁止行为
        forbidden = narrative_context.get("forbidden_actions", [])
        if forbidden:
            parts.append(f"Forbidden: {', '.join(forbidden)}")
        
        return "\n".join(parts)
    
    async def _classify_intent(self, user_text: str) -> str:
        """
        意图分类：判断是否需要专家介入
        
        Returns:
            str: "CHAT" (普通对话) 或 "HARD_TASK" (困难任务)
        """
        # 简单的关键词检测（后续可以用模型）
        hard_task_keywords = [
            "微积分", "积分", "微分", "证明", "推导",
            "代码", "debug", "报错", "error",
            "calculus", "integral", "derivative", "proof"
        ]
        
        text_lower = user_text.lower()
        for keyword in hard_task_keywords:
            if keyword in text_lower:
                return "HARD_TASK"
        
        return "CHAT"
    
    async def _call_expert(self, user_text: str) -> str:
        """
        调用专家模型 (DeepSeek V3)
        
        TODO: Phase 2 实现 MoA
        """
        if self.expert_adapter is None:
            return ""
        
        # result = await self.expert_adapter.solve(user_text)
        return "Expert analysis placeholder"
    
    async def _log_interaction(
        self,
        user_text: str,
        response: str,
        emotion: str
    ) -> None:
        """异步记录交互到记忆系统"""
        try:
            # 只记录重要的交互
            if len(user_text) > 20 or emotion != "neutral":
                await self.narrative_mgr.add_memory(
                    f"User said: '{user_text[:100]}'. AI responded with emotion: {emotion}",
                    memory_type="episodic"
                )
        except Exception as e:
            logger.warning(f"记忆记录失败: {e}")
    
    def set_expert_adapter(self, adapter) -> None:
        """设置专家适配器 (用于 MoA)"""
        self.expert_adapter = adapter
        logger.info("专家适配器已配置")
    
    async def handle_idle(self, silence_duration: float) -> Optional[DirectorDecision]:
        """
        处理用户沉默（主动性循环）
        
        当用户沉默超过一定时间时，AI 可能主动发起对话
        
        Args:
            silence_duration: 沉默时长（秒）
            
        Returns:
            Optional[DirectorDecision]: 主动发起的对话，或 None
        """
        bio_snapshot = self.bio_state.get_snapshot()
        
        # 如果她"无聊"或"担心"，主动发起对话
        if bio_snapshot.dopamine < 40 and silence_duration > 30:
            # 构造主动对话
            brain_response = await self.brain.process(
                user_input="[SYSTEM: User has been silent. Initiate conversation if appropriate.]",
                bio_state={
                    "cortisol": bio_snapshot.cortisol,
                    "dopamine": bio_snapshot.dopamine,
                    "mood": bio_snapshot.current_mood
                },
                temperature=bio_snapshot.temperature
            )
            
            return DirectorDecision(
                response_text=brain_response.response,
                emotion_tag=brain_response.emotion_tag,
                action_hints=brain_response.action_hints,
                inner_monologue=brain_response.inner_monologue,
                triggered_reflex=None,
                llm_temperature=bio_snapshot.temperature
            )
        
        return None

