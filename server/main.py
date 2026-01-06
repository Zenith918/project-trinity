"""
Project Trinity - Server Entry Point
服务端主入口

启动方式:
    uvicorn main:app --host 0.0.0.0 --port 8000

或:
    python main.py
"""

import sys
import os

# ============== 路径黑科技 ==============
# 强制将 server 目录加入路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 强制将 CosyVoice 加入路径 (最高优先级)
COSYVOICE_PATH = "/workspace/CosyVoice"
if os.path.exists(COSYVOICE_PATH):
    # 移除已存在的路径以防止重复，然后插入到最前面
    if COSYVOICE_PATH in sys.path:
        sys.path.remove(COSYVOICE_PATH)
    sys.path.insert(0, COSYVOICE_PATH)
    print(f"✅ 已强制添加 CosyVoice 路径: {COSYVOICE_PATH}")
    
    # 验证是否能导入
    try:
        import cosyvoice
        print(f"✅ CosyVoice 模块验证成功: {cosyvoice.__file__}")
    except ImportError as e:
        print(f"❌ CosyVoice 模块验证失败: {e}")
else:
    print(f"⚠️ 未找到 CosyVoice 目录: {COSYVOICE_PATH}")
# ========================================

import asyncio
from contextlib import asynccontextmanager
from typing import Optional
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger
import uvicorn

from config import settings
from adapters import VoiceAdapter, BrainAdapter, MouthAdapter, DriverAdapter
from mind_engine import BioState, NarrativeManager, EgoDirector


# ============== 全局组件 ==============
voice_adapter: Optional[VoiceAdapter] = None
brain_adapter: Optional[BrainAdapter] = None
mouth_adapter: Optional[MouthAdapter] = None
driver_adapter: Optional[DriverAdapter] = None

bio_state: Optional[BioState] = None
narrative_mgr: Optional[NarrativeManager] = None
ego_director: Optional[EgoDirector] = None


# ============== 生命周期管理 ==============
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global voice_adapter, brain_adapter, mouth_adapter, driver_adapter
    global bio_state, narrative_mgr, ego_director
    
    logger.info("🔮 Project Trinity 启动中...")
    
    # 初始化 Layer 1: 本我 (BioState)
    bio_state = BioState()
    logger.success("✓ Layer 1 (本我) 初始化完成")
    
    # 初始化 Layer 2: 超我 (NarrativeManager)
    narrative_mgr = NarrativeManager(
        qdrant_host=settings.memory.qdrant_host,
        qdrant_port=settings.memory.qdrant_port
    )
    await narrative_mgr.initialize()
    logger.success("✓ Layer 2 (超我) 初始化完成")
    
    # 初始化适配器 (可选，根据环境决定是否加载模型)
    if not settings.server.debug:
        # 生产环境: 加载所有模型
        voice_adapter = VoiceAdapter(
            model_name=settings.model.funasr_model,
            device=settings.model.funasr_device
        )
        
        brain_adapter = BrainAdapter(
            model_path=settings.model.qwen_model_path,
            tensor_parallel_size=settings.model.qwen_tensor_parallel_size
        )
        
        mouth_adapter = MouthAdapter(
            model_path=settings.model.cosyvoice_model_path
        )
        
        driver_adapter = DriverAdapter(
            model_path=settings.model.geneface_model_path
        )
        
        # 并行初始化所有适配器
        results = await asyncio.gather(
            voice_adapter.initialize(),
            brain_adapter.initialize(),
            mouth_adapter.initialize(),
            driver_adapter.initialize(),
            return_exceptions=True
        )
        
        for name, result in zip(
            ["Voice", "Brain", "Mouth", "Driver"],
            results
        ):
            if isinstance(result, Exception):
                logger.error(f"✗ {name} Adapter 初始化失败: {result}")
            elif result:
                logger.success(f"✓ {name} Adapter 初始化完成")
    else:
        logger.warning("⚠ Debug 模式: 跳过模型加载")
        # Debug 模式使用 Mock
        brain_adapter = BrainAdapter()  # 不初始化
    
    # 初始化 Layer 3: 自我 (EgoDirector)
    if brain_adapter:
        ego_director = EgoDirector(
            brain=brain_adapter,
            bio_state=bio_state,
            narrative_mgr=narrative_mgr
        )
        logger.success("✓ Layer 3 (自我) 初始化完成")
    
    logger.info("🎭 Project Trinity 准备就绪!")
    
    yield
    
    # 清理
    logger.info("正在关闭 Project Trinity...")
    
    if voice_adapter:
        await voice_adapter.shutdown()
    if brain_adapter:
        await brain_adapter.shutdown()
    if mouth_adapter:
        await mouth_adapter.shutdown()
    if driver_adapter:
        await driver_adapter.shutdown()
    if narrative_mgr:
        await narrative_mgr.shutdown()
    
    logger.info("Project Trinity 已关闭")


# ============== FastAPI 应用 ==============
app = FastAPI(
    title="Project Trinity",
    description="Next-Gen Digital Life Engine",
    version="0.1.0",
    lifespan=lifespan
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== 数据模型 ==============
class ChatRequest(BaseModel):
    """聊天请求"""
    text: str
    emotion: str = "neutral"
    visual_context: Optional[str] = None


class ChatResponse(BaseModel):
    """聊天响应"""
    response: str
    emotion_tag: str
    action_hints: list
    bio_state: dict


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    components: dict


# ============== API 路由 ==============
@app.get("/")
async def root():
    """根路由"""
    return {
        "name": "Project Trinity",
        "version": "0.1.0",
        "status": "running"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    components = {
        "bio_state": bio_state is not None,
        "narrative_mgr": narrative_mgr is not None and narrative_mgr.is_initialized,
        "ego_director": ego_director is not None,
        "voice_adapter": voice_adapter is not None and voice_adapter.is_initialized if voice_adapter else False,
        "brain_adapter": brain_adapter is not None and brain_adapter.is_initialized if brain_adapter else False,
    }
    
    status = "healthy" if all(components.values()) else "degraded"
    
    return HealthResponse(status=status, components=components)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    文本对话接口
    
    用于测试和简单场景
    """
    if ego_director is None:
        raise HTTPException(status_code=503, detail="EgoDirector 未初始化")
    
    try:
        decision = await ego_director.process(
            user_text=request.text,
            detected_emotion=request.emotion,
            visual_context=request.visual_context
        )
        
        return ChatResponse(
            response=decision.response_text,
            emotion_tag=decision.emotion_tag,
            action_hints=decision.action_hints,
            bio_state={
                "temperature": decision.llm_temperature,
                "triggered_reflex": decision.triggered_reflex
            }
        )
        
    except Exception as e:
        logger.error(f"Chat 处理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket 实时通信
    
    协议:
    - Client -> Server: { "type": "audio", "data": base64 } 或 { "type": "text", "data": "..." }
    - Server -> Client: { "type": "response", "text": "...", "audio": base64, "flame": [...] }
    """
    await websocket.accept()
    logger.info("WebSocket 客户端已连接")
    
    try:
        while True:
            # 接收消息
            data = await websocket.receive_text()
            message = json.loads(data)
            
            msg_type = message.get("type", "text")
            
            if msg_type == "text":
                # 文本消息
                user_text = message.get("data", "")
                emotion = message.get("emotion", "neutral")
                
                if ego_director:
                    decision = await ego_director.process(
                        user_text=user_text,
                        detected_emotion=emotion
                    )
                    
                    response = {
                        "type": "response",
                        "text": decision.response_text,
                        "emotion": decision.emotion_tag,
                        "actions": decision.action_hints,
                        "reflex": decision.triggered_reflex
                    }
                else:
                    response = {
                        "type": "error",
                        "message": "System not ready"
                    }
                
                await websocket.send_text(json.dumps(response))
            
            elif msg_type == "audio":
                # 音频消息 (TODO: Phase 1)
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Audio processing not implemented yet"
                }))
            
            elif msg_type == "heartbeat":
                # 心跳
                await websocket.send_text(json.dumps({
                    "type": "heartbeat",
                    "status": "ok"
                }))
    
    except WebSocketDisconnect:
        logger.info("WebSocket 客户端已断开")
    except Exception as e:
        logger.error(f"WebSocket 错误: {e}")


# ============== 主入口 ==============
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=settings.server.debug
    )

