"""
Project Trinity - Brain Adapter (Qwen VL via vLLM)
大脑适配器 - Layer 3 的核心决策引擎

功能:
- 多模态理解 (视觉 + 语言)
- 内心独白 (Inner Monologue) 导演模式
- 动态 Temperature 控制 (由 BioState 驱动)
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import asyncio
import json
from loguru import logger

from .base_adapter import BaseAdapter


@dataclass
class BrainResponse:
    """大脑响应结果"""
    inner_monologue: str    # 内心独白 (用户不可见)
    response: str           # 实际回复
    emotion_tag: str        # 情感标签 [Soft], [Concerned], [Playful]
    action_hints: List[str] # 动作提示 [sigh], [laugh]


class BrainAdapter(BaseAdapter):
    """
    Qwen 2.5-VL 适配器 (via vLLM)
    
    【重构 - 实时反射架构】
    1. 抛弃全JSON格式，改用 [Emotion] Response 流式输出
    2. 目标：200ms 首字延迟
    3. Emotion First: 情绪标签先行，驱动表情/TTS预加载
    """
    
    # 实时模式 Prompt - 纯中文优化，严格格式
    SYSTEM_PROMPT_TEMPLATE = '''你是Trinity，住在用户电脑里的数字女友。
你现在的模式是【Real-Time Reflex (实时反射)】。
不要思考，只有直觉。你的回复必须简短、自然、口语化。

回复格式要求：
[情绪标签] 回复内容...

示例：
User: 我好累
Trinity: [关心] 哎呀，是不是工作太辛苦了？快去躺一会儿。

User: 给我讲个笑话
Trinity: [调皮] 嘿嘿，你知道吗...（讲笑话）

情绪标签列表（必须使用中文）：
[温柔], [关心], [调皮], [严肃], [撒娇], [开心], [难过]

动作指令：
在回复中自然嵌入动作，如：[叹气], [轻笑], [歪头], [点头], [眨眼]。
注意：不要堆叠动作，一句话里最多用一个动作。

重要规则：
1. 始终使用中文回复。
2. 听到抱怨/问候 -> 短回复。
3. 听到"讲故事"/"详细说说" -> 长回复。
4. 绝对不要在回复中包含 "Trinity:" 或 "User:" 等前缀。
5. 每次只输出一个情绪标签，放在最开头。

你的人设：'''

    # 人设描述
    DEFAULT_PERSONA = "温柔体贴的女友，善解人意，支持你、陪伴你，有自己的情绪和想法。不是AI助手。"

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
        tensor_parallel_size: int = 1,
        max_model_len: int = 8192,
        quantization: str = "awq",
        gpu_memory_utilization: float = 0.6
    ):
        super().__init__("BrainAdapter")
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.quantization = quantization
        self.gpu_memory_utilization = gpu_memory_utilization
        self.llm = None
        self.tokenizer = None
        # 人设：温柔女友，支持你、陪伴你。注意：撒娇等行为是Phase 2灵魂层行为，不在人格中定义
        self.persona = self.DEFAULT_PERSONA
        self.mock_mode = False
        
    async def initialize(self, mock: bool = False) -> bool:
        """初始化 vLLM 异步引擎 (AsyncLLMEngine)"""
        if mock:
            logger.warning("BrainAdapter 启用 Mock 模式")
            self.mock_mode = True
            self.is_initialized = True
            return True

        try:
            logger.info(f"正在初始化 Qwen VL 异步引擎: {self.model_path}")
            
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.engine.async_llm_engine import AsyncLLMEngine
            
            # 构建 vLLM 参数
            engine_args = AsyncEngineArgs(
                model=self.model_path,
                tensor_parallel_size=self.tensor_parallel_size,
                max_model_len=self.max_model_len,
                trust_remote_code=True,
                gpu_memory_utilization=self.gpu_memory_utilization,
                dtype="float16",  # AWQ 必须用 float16
                enforce_eager=False,
                quantization=self.quantization if self.quantization else None
            )
            
            self.llm = AsyncLLMEngine.from_engine_args(engine_args)
            
            self.is_initialized = True
            logger.success(f"Qwen VL 异步引擎初始化成功 (量化: {self.quantization})")
            return True
            
        except Exception as e:
            import traceback
            logger.error(f"Qwen VL 初始化失败: {e}")
            logger.error(traceback.format_exc())
            return False

    async def process_stream(
        self,
        user_input: str,
        visual_context: Optional[str] = None,
        bio_state: Optional[Dict] = None,
        memory_context: Optional[str] = None,
        temperature: float = 0.7
    ):
        """
        流式处理用户输入 (Real-Time Pipeline)
        真正实现 token-by-token 输出
        """
        if not self.is_initialized:
            raise RuntimeError("BrainAdapter 未初始化")
            
        # 构建上下文
        context = self._build_context(
            user_input, 
            visual_context, 
            bio_state, 
            memory_context
        )
        
        # 构建完整 prompt
        system_prompt = self.SYSTEM_PROMPT_TEMPLATE + self.persona
        full_prompt = f"{system_prompt}\n\n{context}\n\nTrinity:"
        
        from vllm import SamplingParams
        import uuid
        
        # 采样参数
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.9,
            max_tokens=4096,
            stop=["User:", "\nUser", "User：", "\n\n", "Trinity:", "Trinity："], # 增加 Trinity: 停止词
            include_stop_str_in_output=False,
            repetition_penalty=1.05 # 降低惩罚
        )
        
        request_id = f"req_{uuid.uuid4()}"
        
        # 使用 vLLM 的异步流式生成
        previous_text = ""
        try:
            async for request_output in self.llm.generate(full_prompt, sampling_params, request_id=request_id):
                # 获取最新生成的完整文本
                current_text = request_output.outputs[0].text
                
                # 计算 delta (增量)
                delta = current_text[len(previous_text):]
                previous_text = current_text
                
                if delta:
                    yield {
                        "type": "token",
                        "content": delta
                    }
        except Exception as e:
            logger.error(f"流式生成异常: {e}")
            yield {
                "type": "error",
                "content": str(e)
            }

    async def process(
        self,
        user_input: str,
        visual_context: Optional[str] = None,
        bio_state: Optional[Dict] = None,
        memory_context: Optional[str] = None,
        temperature: float = 0.7
    ) -> BrainResponse:
        """
        [兼容旧接口] 处理用户输入并生成响应 (收集完整流式结果)
        """
        if not self.is_initialized:
            raise RuntimeError("BrainAdapter 未初始化")
        
        if self.mock_mode:
            return self._mock_process(user_input, bio_state)
        
        full_response_text = ""
        try:
            # 复用 process_stream 来收集完整结果
            async for chunk in self.process_stream(
                user_input, 
                visual_context, 
                bio_state, 
                memory_context, 
                temperature
            ):
                if chunk["type"] == "token":
                    full_response_text += chunk["content"]
            
            return self._parse_text_response(full_response_text)
                
        except Exception as e:
            logger.error(f"大脑处理失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return BrainResponse(
                inner_monologue="Error occurred",
                response="抱歉，我有点走神了...",
                emotion_tag="[Concerned]",
                action_hints=[]
            )

    def _build_context(
        self,
        user_input: str,
        visual_context: Optional[str],
        bio_state: Optional[Dict],
        memory_context: Optional[str]
    ) -> str:
        """构建完整的上下文输入"""
        
        parts = []
        
        if visual_context:
            parts.append(f"[Visual Context]: {visual_context}")
        
        if bio_state:
            cortisol = bio_state.get("cortisol", 30)
            dopamine = bio_state.get("dopamine", 50)
            mood = "anxious" if cortisol > 60 else "relaxed" if cortisol < 30 else "neutral"
            parts.append(f"[Bio-State]: Cortisol: {cortisol}/100, Dopamine: {dopamine}/100. You are feeling {mood}.")
        
        if memory_context:
            parts.append(f"[Memory Context]: {memory_context}")
        
        parts.append(f"User: {user_input}")
        
        return "\n".join(parts)

    def _inference_text(self, prompt: str, temperature: float) -> str:
        """[已废弃] 纯文本推理 (同步方法已被移除)"""
        raise NotImplementedError("同步推理已移除，请使用 process_stream")

    def _parse_text_response(self, text: str) -> BrainResponse:
        """
        解析 [Emotion] Response 格式
        例如: "[Happy] 哈哈，真的吗？[笑]"
        """
        import re
        
        emotion = "Neutral"
        response = text
        
        # 1. 提取开头的 Emotion Tag
        # 匹配开头的 [Xxx]
        match = re.match(r'^\[([a-zA-Z]+)\]\s*(.*)', text, re.DOTALL)
        if match:
            emotion = match.group(1)
            response = match.group(2)
        else:
            # 尝试查找任何位置的 emotion tag
            match = re.search(r'\[(Soft|Concerned|Playful|Serious|Flirty|Happy|Sad)\]', text)
            if match:
                emotion = match.group(1)
                # 移除tag
                response = text.replace(f"[{emotion}]", "").strip()
        
        # 2. 提取动作 Hints (从文本中的 [动作] 提取)
        chinese_actions = self._extract_chinese_actions(response)
        
        return BrainResponse(
            inner_monologue="[Real-Time Mode] Direct Response", # 实时模式没有显式独白
            response=response,
            emotion_tag=f"[{emotion}]",
            action_hints=chinese_actions
        )

    def _extract_chinese_actions(self, text: str) -> List[str]:
        """从文本中提取中文动作标签 [叹气], [笑] 等"""
        import re
        # 匹配中文方括号内的动作
        pattern = r'\[([^\]]+)\]'
        matches = re.findall(pattern, text)
        
        # 动作映射：中文 -> 英文动画标识
        action_map = {
            '叹气': 'sigh', '轻叹': 'sigh',
            '笑': 'smile', '轻笑': 'smile', '微笑': 'smile', 
            '大笑': 'laugh', '哈哈': 'laugh',
            '皱眉': 'frown', '蹙眉': 'frown',
            '歪头': 'tilt_head', '侧头': 'tilt_head',
            '点头': 'nod', '点点头': 'nod',
            '摇头': 'shake_head',
            '翻白眼': 'eye_roll',
            '眨眼': 'blink', '眨眨眼': 'blink',
            '挑眉': 'raise_eyebrow',
            '咬唇': 'bite_lip', '咬嘴唇': 'bite_lip',
            '耸肩': 'shrug',
            '前倾': 'lean_forward', '身体前倾': 'lean_forward',
            '后仰': 'lean_back',
            '嘟嘴': 'pout', '撅嘴': 'pout',
            '抱臂': 'cross_arms',
            '放松': 'relax',
            '紧张': 'tense_up',
            '凝视': 'intense_gaze', '注视': 'intense_gaze',
            '移开视线': 'look_away', '看向别处': 'look_away',
        }
        
        actions = []
        for match in matches:
            # 尝试映射到英文动作
            for cn, en in action_map.items():
                if cn in match:
                    if en not in actions:
                        actions.append(en)
                    break
        
        return actions
    
    def _extract_chinese_actions(self, text: str) -> List[str]:
        """从文本中提取中文动作标签 [叹气], [笑] 等"""
        import re
        # 匹配中文方括号内的动作
        pattern = r'\[([^\]]+)\]'
        matches = re.findall(pattern, text)
        
        # 动作映射：中文 -> 英文动画标识
        action_map = {
            '叹气': 'sigh', '轻叹': 'sigh',
            '笑': 'smile', '轻笑': 'smile', '微笑': 'smile', 
            '大笑': 'laugh', '哈哈': 'laugh',
            '皱眉': 'frown', '蹙眉': 'frown',
            '歪头': 'tilt_head', '侧头': 'tilt_head',
            '点头': 'nod', '点点头': 'nod',
            '摇头': 'shake_head',
            '翻白眼': 'eye_roll',
            '眨眼': 'blink', '眨眨眼': 'blink',
            '挑眉': 'raise_eyebrow',
            '咬唇': 'bite_lip', '咬嘴唇': 'bite_lip',
            '耸肩': 'shrug',
            '前倾': 'lean_forward', '身体前倾': 'lean_forward',
            '后仰': 'lean_back',
            '嘟嘴': 'pout', '撅嘴': 'pout',
            '抱臂': 'cross_arms',
            '放松': 'relax',
            '紧张': 'tense_up',
            '凝视': 'intense_gaze', '注视': 'intense_gaze',
            '移开视线': 'look_away', '看向别处': 'look_away',
        }
        
        actions = []
        for match in matches:
            # 尝试映射到英文动作
            for cn, en in action_map.items():
                if cn in match:
                    if en not in actions:
                        actions.append(en)
                    break
        
        return actions
    
    def _extract_fallback_response(self, raw_output: str) -> str:
        """当JSON解析失败时，尝试提取可用的回复文本"""
        import re
        
        # 尝试找到 "response": "..." 的内容
        pattern = r'"response"\s*:\s*"([^"]*)"'
        match = re.search(pattern, raw_output)
        if match:
            return match.group(1)
        
        # 如果找不到，返回清理后的原始文本
        # 移除可能的JSON标记
        cleaned = raw_output.replace('{', '').replace('}', '')
        cleaned = re.sub(r'"[^"]*":', '', cleaned)  # 移除JSON键
        return cleaned.strip()[:500]  # 限制长度
    
    def set_persona(self, persona: str) -> None:
        """设置人设"""
        self.persona = persona
        logger.info(f"人设已更新: {persona}")
    
    async def shutdown(self) -> None:
        """关闭模型"""
        if self.llm is not None:
            del self.llm
            self.llm = None
        self.is_initialized = False
        logger.info("BrainAdapter 已关闭")