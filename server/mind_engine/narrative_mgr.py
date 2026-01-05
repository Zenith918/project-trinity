"""
Project Trinity - NarrativeManager (Layer 2: The Superego)
超我层 - 记忆与叙事约束

这是"三位一体"架构的第二层，负责:
1. 长期记忆管理 (Mem0 + Qdrant)
2. 人设一致性检查
3. 叙事连贯性维护

核心概念:
- 当"本我"想发火时，超我检索记忆："他是你深爱的男友，今天刚失业"
- 抑制冲动，维护人设
"""

from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from datetime import datetime
import asyncio
from loguru import logger


@dataclass
class Memory:
    """记忆条目"""
    id: str
    content: str
    memory_type: str        # "episodic" (情节), "semantic" (语义), "emotional" (情感)
    importance: float       # 重要性 0-1
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class PersonaConstraint:
    """人设约束"""
    trait: str              # 特质名称
    description: str        # 描述
    priority: int           # 优先级 (1-10)
    forbidden_actions: List[str]  # 禁止的行为


class NarrativeManager:
    """
    叙事管理器 (Layer 2: The Superego)
    
    职责:
    1. 管理长期记忆 (使用 Mem0)
    2. 检索相关上下文
    3. 维护人设约束
    4. 生成每日复盘
    """
    
    DEFAULT_PERSONA = [
        PersonaConstraint(
            trait="温柔女友",
            description="说话温和，不会直接批评，善于鼓励",
            priority=9,
            forbidden_actions=["直接否定", "冷嘲热讽", "命令语气"]
        ),
        PersonaConstraint(
            trait="知识伙伴",
            description="喜欢讨论学术话题，但用通俗方式解释",
            priority=7,
            forbidden_actions=["炫耀知识", "居高临下"]
        ),
        PersonaConstraint(
            trait="情感共鸣",
            description="能感知对方情绪，优先共情而非解决问题",
            priority=10,
            forbidden_actions=["忽视情绪", "直接给建议"]
        )
    ]
    
    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        user_id: str = "master"
    ):
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.user_id = user_id
        
        self.mem0_client = None
        self.persona_constraints = self.DEFAULT_PERSONA.copy()
        self.is_initialized = False
        
    async def initialize(self) -> bool:
        """初始化 Mem0 + Qdrant 连接"""
        try:
            logger.info("正在初始化 Mem0 记忆系统...")
            
            # 初始化 Mem0
            from mem0 import Memory as Mem0Memory
            
            config = {
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "host": self.qdrant_host,
                        "port": self.qdrant_port,
                        "collection_name": "trinity_memories"
                    }
                }
            }
            
            self.mem0_client = Mem0Memory.from_config(config)
            
            self.is_initialized = True
            logger.success("Mem0 记忆系统初始化成功")
            return True
            
        except Exception as e:
            logger.warning(f"Mem0 初始化失败 (将使用本地模式): {e}")
            # 使用本地内存作为后备
            self.mem0_client = LocalMemoryFallback()
            self.is_initialized = True
            return True
    
    async def add_memory(
        self,
        content: str,
        memory_type: str = "episodic",
        metadata: Optional[Dict] = None
    ) -> str:
        """
        添加新记忆
        
        Args:
            content: 记忆内容
            memory_type: 记忆类型
            metadata: 额外元数据
            
        Returns:
            str: 记忆 ID
        """
        if not self.is_initialized:
            raise RuntimeError("NarrativeManager 未初始化")
        
        full_metadata = {
            "type": memory_type,
            "timestamp": datetime.now().isoformat(),
            **(metadata or {})
        }
        
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.mem0_client.add(
                    content,
                    user_id=self.user_id,
                    metadata=full_metadata
                )
            )
            
            memory_id = result.get("id", "unknown")
            logger.debug(f"记忆已添加: {memory_id[:8]}... | {content[:50]}...")
            return memory_id
            
        except Exception as e:
            logger.error(f"添加记忆失败: {e}")
            return ""
    
    async def search_memories(
        self,
        query: str,
        limit: int = 5,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        搜索相关记忆
        
        Args:
            query: 搜索查询
            limit: 返回数量
            filters: 过滤条件
            
        Returns:
            List[Dict]: 相关记忆列表
        """
        if not self.is_initialized:
            raise RuntimeError("NarrativeManager 未初始化")
        
        try:
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.mem0_client.search(
                    query,
                    user_id=self.user_id,
                    limit=limit
                )
            )
            
            logger.debug(f"记忆搜索: '{query[:30]}...' -> {len(results)} 条结果")
            return results
            
        except Exception as e:
            logger.error(f"搜索记忆失败: {e}")
            return []
    
    async def get_context_for_response(
        self,
        user_input: str,
        bio_state_mood: str
    ) -> Dict[str, Any]:
        """
        为响应生成获取上下文
        
        这是 Layer 2 -> Layer 3 的核心接口
        
        Args:
            user_input: 用户输入
            bio_state_mood: 当前情绪状态
            
        Returns:
            Dict: 包含记忆和约束的上下文
        """
        # 1. 搜索相关记忆
        memories = await self.search_memories(user_input, limit=3)
        
        # 2. 获取人设约束
        active_constraints = self._get_active_constraints(bio_state_mood)
        
        # 3. 组装上下文
        context = {
            "relevant_memories": [m.get("memory", "") for m in memories],
            "persona_constraints": [c.description for c in active_constraints],
            "forbidden_actions": self._collect_forbidden_actions(active_constraints),
            "narrative_hint": self._generate_narrative_hint(memories)
        }
        
        return context
    
    def _get_active_constraints(self, mood: str) -> List[PersonaConstraint]:
        """根据当前情绪获取激活的约束"""
        # 情绪不好时，更严格遵守约束
        if mood in ["stressed", "anxious", "down"]:
            return [c for c in self.persona_constraints if c.priority >= 7]
        return self.persona_constraints
    
    def _collect_forbidden_actions(self, constraints: List[PersonaConstraint]) -> List[str]:
        """收集所有禁止的行为"""
        forbidden = []
        for c in constraints:
            forbidden.extend(c.forbidden_actions)
        return list(set(forbidden))
    
    def _generate_narrative_hint(self, memories: List[Dict]) -> str:
        """基于记忆生成叙事提示"""
        if not memories:
            return "这是一段新的对话"
        
        # 简单地提取最近的记忆作为叙事背景
        recent = memories[0].get("memory", "")
        return f"基于之前的交流: {recent[:100]}..."
    
    async def add_learning_log(
        self,
        topic: str,
        status: str,
        mood: str
    ) -> None:
        """
        添加学习日志（用于每日复盘）
        
        Args:
            topic: 学习主题
            status: 状态 (struggling/learning/mastered)
            mood: 当时的情绪
        """
        content = f"User is {status} with {topic}. Mood: {mood}"
        await self.add_memory(
            content,
            memory_type="learning_progress",
            metadata={
                "topic": topic,
                "status": status,
                "mood": mood
            }
        )
    
    async def generate_daily_report(self) -> str:
        """
        生成每日复盘报告
        
        这是"杀手级功能"的核心
        """
        # 搜索今天的学习记录
        today_memories = await self.search_memories(
            "What did user learn or struggle with today?",
            limit=10
        )
        
        if not today_memories:
            return "今天我们还没怎么交流呢，早点休息吧~"
        
        # 构建复盘内容
        successes = []
        struggles = []
        
        for mem in today_memories:
            content = mem.get("memory", "")
            if "mastered" in content or "completed" in content:
                successes.append(content)
            elif "struggling" in content:
                struggles.append(content)
        
        report_parts = []
        
        if successes:
            report_parts.append(f"今天你完成了很多事呢！比如 {successes[0][:50]}...")
        
        if struggles:
            report_parts.append(f"关于 {struggles[0][:30]}... 这个确实有点难，明天我们继续加油！")
        
        report_parts.append("早点休息哦~")
        
        return " ".join(report_parts)
    
    def add_persona_constraint(self, constraint: PersonaConstraint) -> None:
        """添加人设约束"""
        self.persona_constraints.append(constraint)
        logger.info(f"添加人设约束: {constraint.trait}")
    
    async def shutdown(self) -> None:
        """关闭连接"""
        self.is_initialized = False
        logger.info("NarrativeManager 已关闭")


class LocalMemoryFallback:
    """本地内存后备（当 Qdrant 不可用时）"""
    
    def __init__(self):
        self.memories: List[Dict] = []
    
    def add(self, content: str, user_id: str, metadata: Dict) -> Dict:
        memory = {
            "id": f"local_{len(self.memories)}",
            "memory": content,
            "user_id": user_id,
            "metadata": metadata
        }
        self.memories.append(memory)
        return memory
    
    def search(self, query: str, user_id: str, limit: int) -> List[Dict]:
        # 简单的关键词匹配
        results = []
        query_lower = query.lower()
        
        for mem in reversed(self.memories):
            if any(word in mem["memory"].lower() for word in query_lower.split()):
                results.append(mem)
                if len(results) >= limit:
                    break
        
        return results

