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
        gpu_memory_utilization: float = 0.6,
        remote_url: Optional[str] = None
    ):
        super().__init__("BrainAdapter")
        self.model_path = model_path
        self.remote_url = remote_url
        self.remote_mode = False
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.quantization = quantization
        self.gpu_memory_utilization = gpu_memory_utilization
        self.llm = None
        self.tokenizer = None
        self.persona = self.DEFAULT_PERSONA
        self.mock_mode = False
        
    async def initialize(self, mock: bool = False) -> bool:
        """初始化适配器"""
        if self.model_path == "REMOTE":
            self.remote_mode = True
            logger.info(f"BrainAdapter 运行在远程模式: {self.remote_url}")
            self.is_initialized = True
            return True

        if mock:
            logger.warning("BrainAdapter 启用 Mock 模式")
            self.mock_mode = True
            self.is_initialized = True
            return True

        try:
            logger.info(f"正在初始化 Qwen VL 异步引擎: {self.model_path}")
            
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.engine.async_llm_engine import AsyncLLMEngine
            
            engine_args = AsyncEngineArgs(
                model=self.model_path,
                tensor_parallel_size=self.tensor_parallel_size,
                max_model_len=self.max_model_len,
                trust_remote_code=True,
                gpu_memory_utilization=self.gpu_memory_utilization,
                dtype="float16",
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
        """流式处理用户输入"""
        if not self.is_initialized:
            raise RuntimeError("BrainAdapter 未初始化")

        context = self._build_context(
            user_input, 
            visual_context, 
            bio_state, 
            memory_context
        )
        
        system_prompt = self.SYSTEM_PROMPT_TEMPLATE + self.persona
        full_prompt = f"{system_prompt}\n\n{context}\n\nTrinity:"
        
        # 远程模式处理 - 真正的流式传输 (SSE)
        if self.remote_mode:
            import aiohttp
        import json
            async with aiohttp.ClientSession() as session:
                try:
                    payload = {"prompt": full_prompt, "max_tokens": 256, "temperature": temperature}
                    # 使用流式端点
                    async with session.post(
                        f"{self.remote_url}/chat/stream", 
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=60)
                    ) as resp:
                        if resp.status == 200:
                            # 消费 SSE 流
                            async for line in resp.content:
                                line = line.decode('utf-8').strip()
                                if line.startswith('data: '):
                                    try:
                                        data = json.loads(line[6:])
                                        if 'token' in data:
                                            yield {"type": "token", "content": data['token']}
                                        elif 'done' in data:
                                            break
                                        elif 'error' in data:
                                            yield {"type": "error", "content": data['error']}
                                    except json.JSONDecodeError:
                                        continue
                        else:
                            yield {"type": "error", "content": f"Remote Error: {resp.status}"}
                except aiohttp.ClientError as e:
                    logger.error(f"Remote Brain Connection Error: {e}")
                    yield {"type": "error", "content": str(e)}
                except Exception as e:
                    logger.error(f"Remote Brain Error: {e}")
                    yield {"type": "error", "content": str(e)}
            return

        # 本地模式处理
        if self.mock_mode:
            # Mock
            yield {"type": "token", "content": "[Mock] I hear you."}
            return

        from vllm import SamplingParams
        import uuid
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.9,
            max_tokens=4096,
            stop=["User:", "\nUser", "User：", "\n\n", "Trinity:", "Trinity："],
            include_stop_str_in_output=False,
            repetition_penalty=1.05
        )
        
        request_id = f"req_{uuid.uuid4()}"
        previous_text = ""
        
        try:
            async for request_output in self.llm.generate(full_prompt, sampling_params, request_id=request_id):
                current_text = request_output.outputs[0].text
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
        """[兼容旧接口] 处理用户输入并生成响应"""
        if not self.is_initialized:
            raise RuntimeError("BrainAdapter 未初始化")
        
        full_response_text = ""
        try:
            async for chunk in self.process_stream(
                user_input, 
                visual_context, 
                bio_state, 
                memory_context, 
                temperature
            ):
                if chunk.get("type") == "token":
                    full_response_text += chunk["content"]
            
            return self._parse_text_response(full_response_text)
                
        except Exception as e:
            logger.error(f"大脑处理失败: {e}")
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

    def _parse_text_response(self, text: str) -> BrainResponse:
        import re
        emotion = "Neutral"
        response = text
        match = re.match(r'^\[([a-zA-Z]+)\]\s*(.*)', text, re.DOTALL)
        if match:
            emotion = match.group(1)
            response = match.group(2)
        else:
            match = re.search(r'\[(Soft|Concerned|Playful|Serious|Flirty|Happy|Sad)\]', text)
            if match:
                emotion = match.group(1)
                response = text.replace(f"[{emotion}]", "").strip()
        
        chinese_actions = self._extract_chinese_actions(response)
        
        return BrainResponse(
            inner_monologue="[Real-Time Mode] Direct Response",
            response=response,
            emotion_tag=f"[{emotion}]",
            action_hints=chinese_actions
        )

    def _extract_chinese_actions(self, text: str) -> List[str]:
        import re
        pattern = r'\[([^\]]+)\]'
        matches = re.findall(pattern, text)
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
            for cn, en in action_map.items():
                if cn in match:
                    if en not in actions:
                        actions.append(en)
                    break
        return actions
    
    def set_persona(self, persona: str) -> None:
        self.persona = persona
        logger.info(f"人设已更新: {persona}")
    
    async def shutdown(self) -> None:
        if self.llm is not None:
            del self.llm
            self.llm = None
        self.is_initialized = False
        logger.info("BrainAdapter 已关闭")
