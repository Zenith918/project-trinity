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
import time
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from loguru import logger
import uvicorn
import shutil
import uuid
from pathlib import Path

from config import settings
from adapters import VoiceAdapter, BrainAdapter, MouthAdapter, DriverAdapter
from mind_engine import BioState, NarrativeManager, EgoDirector
from monitor import SystemMonitor

def write_chat_log(log_data: dict):
    """
    对话日志 - 按天归档到 logs/conversations/YYYY-MM-DD.jsonl
    """
    try:
        import json
        from pathlib import Path
        
        # 计算统计数据
        total_time_s = time.time() - log_data["start"]
        speed = len(log_data["output"]) / total_time_s if total_time_s > 0 else 0
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user": log_data["input"],
            "assistant": log_data["output"],
            "metrics": {
                "ttft_ms": round(log_data.get("ttft", 0), 2),
                "total_time_s": round(total_time_s, 2),
                "speed_char_per_s": round(speed, 2)
            }
        }
        
        # 按天归档
        log_dir = Path("/workspace/project-trinity/project-trinity/logs/conversations")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        today = datetime.now().strftime("%Y-%m-%d")
        log_path = log_dir / f"{today}.jsonl"
        
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info(f"📝 对话已记录: {today}.jsonl ({len(entry['assistant'])} chars)")
            
    except Exception as e:
        logger.error(f"对话日志写入失败: {e}")

# ============== 全局组件 ==============
monitor: Optional[SystemMonitor] = None
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
    global bio_state, narrative_mgr, ego_director, monitor
    
    # 启动资源监控
    monitor = SystemMonitor()
    monitor.start()

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
        # 检查是否为微服务模式
        if os.getenv("TRINITY_MODE") == "microservice":
            # ============== 微服务模式 ==============
            # Brain 和 Mouth 通过远程 Cortex 服务器访问
            logger.info("🚀 微服务模式: 连接到 Cortex Model Server...")
            cortex_url = os.getenv("CORTEX_URL", "http://localhost:9000")
            
            # 1. Remote Brain
            try:
                logger.info(f"正在初始化 Remote BrainAdapter -> {cortex_url}/brain...")
                brain_adapter = BrainAdapter(
                    model_path="REMOTE",
                    remote_url=f"{cortex_url}/brain"
                )
                await brain_adapter.initialize()
                logger.success("✓ Remote BrainAdapter 初始化完成")
            except Exception as e:
                logger.error(f"✗ Remote BrainAdapter 失败: {e}")

            # 2. Remote Mouth
            try:
                logger.info(f"正在初始化 Remote MouthAdapter -> {cortex_url}/mouth...")
                mouth_adapter = MouthAdapter(
                    model_path="REMOTE",
                    remote_url=f"{cortex_url}/mouth"
                )
                await mouth_adapter.initialize()
                logger.success("✓ Remote MouthAdapter 初始化完成")
            except Exception as e:
                logger.error(f"✗ Remote MouthAdapter 失败: {e}")
                
            # 3. Voice (本地，FunASR 较轻)
            try:
                logger.info("正在初始化 VoiceAdapter (Local)...")
                voice_adapter = VoiceAdapter(
                    model_name=settings.model.funasr_model,
                    device=settings.model.funasr_device
                )
                await voice_adapter.initialize()
                logger.success("✓ Voice Adapter 初始化完成")
            except Exception as e:
                logger.error(f"✗ Voice Adapter 失败: {e}")
    
            # 4. Driver (本地)
            try:
                logger.info("正在初始化 DriverAdapter (Local)...")
                driver_adapter = DriverAdapter(
                    geneface_path=settings.model.geneface_model_path
                )
                await driver_adapter.initialize()
                logger.success("✓ Driver Adapter 初始化完成")
            except Exception as e:
                logger.error(f"✗ Driver Adapter 失败: {e}")
                
        else:
            # ============== 单体模式 ==============
            # 所有模型在本地加载
            logger.info("--- 开始串行初始化组件 (单体模式) ---")
            
            # 1. 大脑 (Qwen) - 最吃显存，必须第一个加载
            try:
                logger.info("正在初始化 BrainAdapter (Priority 1)...")
                brain_adapter = BrainAdapter(
                    model_path=settings.model.qwen_model_path,
                    tensor_parallel_size=settings.model.qwen_tensor_parallel_size,
                    max_model_len=settings.model.qwen_max_model_len,
                    quantization=settings.model.qwen_quantization,
                    gpu_memory_utilization=settings.model.qwen_gpu_memory_utilization
                )
                await brain_adapter.initialize()
                if not brain_adapter.is_initialized:
                    raise RuntimeError("BrainAdapter 初始化失败")
                logger.success("✓ Brain Adapter 初始化完成")
            except Exception as e:
                logger.error(f"✗ Brain Adapter 失败: {e}")

            # 2. 嘴巴 (CosyVoice) - 显存占用第二
            try:
                logger.info("正在初始化 MouthAdapter (Priority 2)...")
                mouth_adapter = MouthAdapter(
                    model_path=settings.model.cosyvoice_model_path
                )
                await mouth_adapter.initialize()
                logger.success("✓ Mouth Adapter 初始化完成")
            except Exception as e:
                logger.error(f"✗ Mouth Adapter 失败: {e}")

            # 3. 听觉 (SenseVoice)
            try:
                logger.info("正在初始化 VoiceAdapter...")
                voice_adapter = VoiceAdapter(
                    model_name=settings.model.funasr_model,
                    device=settings.model.funasr_device
                )
                await voice_adapter.initialize()
                logger.success("✓ Voice Adapter 初始化完成")
            except Exception as e:
                logger.error(f"✗ Voice Adapter 失败: {e}")
                
            # 4. 表情 (GeneFace)
            try:
                logger.info("正在初始化 DriverAdapter...")
                driver_adapter = DriverAdapter(
                    geneface_path=settings.model.geneface_model_path
                )
                await driver_adapter.initialize()
                logger.success("✓ Driver Adapter 初始化完成")
            except Exception as e:
                logger.error(f"✗ Driver Adapter 失败: {e}")
        
    else:
        logger.warning("⚠ Debug 模式: 跳过模型加载")
        # Debug 模式使用 Mock
        brain_adapter = BrainAdapter()
        await brain_adapter.initialize(mock=True)
    
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
    
    if monitor:
        monitor.stop()

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

# 挂载 Web 客户端 (LLM Workbench)
from fastapi.staticfiles import StaticFiles
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "client/llm_workbench")
if os.path.exists(static_dir):
    app.mount("/workbench", StaticFiles(directory=static_dir, html=True), name="workbench")
    logger.info(f"Workbench mounted at /workbench -> {static_dir}")

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest, background_tasks: BackgroundTasks):
    """
    流式对话接口 (Real-Time Reflex)
    直接连接 BrainAdapter，绕过 EgoDirector 的部分逻辑以测试极致速度
    """
    if not brain_adapter:
        raise HTTPException(status_code=503, detail="BrainAdapter 未初始化")
        
    logger.info(f"Stream Request: {request.text[:50]}...")
    
    # 准备日志数据容器（可变对象）
    log_data = {
        "input": request.text,
        "output": "",
        "ttft": 0,
        "start": time.time()
    }
    
    # 添加后台任务，在响应结束后执行
    background_tasks.add_task(write_chat_log, log_data)
    
    async def event_generator():
        # 记录开始时间
        start_time = log_data["start"]
        first_token_sent = False
        
        try:
            # 直接调用 BrainAdapter 的流式方法
            generator = brain_adapter.process_stream(
                user_input=request.text,
                temperature=0.7 
            )
            
            async for chunk in generator:
                if chunk["type"] == "token":
                    content = chunk["content"]
                    log_data["output"] += content # 实时更新日志容器
                    
                    yield content
                    
                    if not first_token_sent:
                        first_token_sent = True
                        ttft_ms = (time.time() - start_time) * 1000
                        log_data["ttft"] = ttft_ms
                        logger.info(f"⚡ Stream TTFT: {ttft_ms:.2f}ms")
                        
                elif chunk["type"] == "error":
                    error_msg = f"[ERROR: {chunk['content']}]"
                    log_data["output"] += error_msg
                    yield error_msg
                    
        except Exception as e:
            logger.error(f"Stream Error: {e}")
            yield f"[SYSTEM ERROR: {str(e)}]"
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

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


@app.post("/avatar/generate")
async def generate_avatar(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...)
):
    """
    [FastAvatar] 从照片生成 3DGS 资产
    这是一个耗时操作，将在后台运行。
    """
    if not driver_adapter:
        raise HTTPException(status_code=503, detail="DriverAdapter 未初始化")
        
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    
    file_ext = image.filename.split(".")[-1]
    file_id = str(uuid.uuid4())
    image_path = upload_dir / f"{file_id}.{file_ext}"
    output_dir = Path("assets/avatars") / file_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
        
    logger.info(f"收到 Avatar 生成请求: {image.filename} -> {file_id}")
    
    # 异步执行生成任务
    async def _run_generation():
        success = await driver_adapter.generate_avatar(str(image_path), str(output_dir))
        if success:
            logger.success(f"Avatar 生成完成: {file_id}")
            # TODO: 通知客户端或更新数据库
        else:
            logger.error(f"Avatar 生成失败: {file_id}")

    background_tasks.add_task(_run_generation)
    
    return {
        "status": "processing", 
        "task_id": file_id,
        "message": "Avatar 生成任务已提交，请稍候。"
    }


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


# ============== 测试端点 ==============

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    语音识别接口 (ASR)
    
    接受音频文件，返回识别文本和情感
    """
    if not voice_adapter or not voice_adapter.is_initialized:
        raise HTTPException(status_code=503, detail="VoiceAdapter 未初始化")
    
    try:
        import io
        import wave
        import numpy as np
        
        # 读取上传的音频文件
        audio_bytes = await file.read()
        
        # 尝试解析 WAV 格式
        try:
            wav_buffer = io.BytesIO(audio_bytes)
            with wave.open(wav_buffer, 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                audio_data = wav_file.readframes(n_frames)
                # 转换为 numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception:
            # 如果不是标准 WAV，尝试直接作为 PCM 处理
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            sample_rate = 16000
        
        # 调用 ASR
        result = await voice_adapter.process(audio_array, sample_rate)
        
        return {
            "text": result.text,
            "emotion": result.emotion,
            "confidence": result.emotion_confidence,
            "language": result.language
        }
        
    except Exception as e:
        logger.error(f"语音识别失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/synthesize")
async def synthesize_speech(request: dict):
    """
    语音合成接口 (TTS)
    
    接受文本，返回音频数据
    """
    if not mouth_adapter:
        raise HTTPException(status_code=503, detail="MouthAdapter 未初始化")
    
    text = request.get("text", "")
    instruct_text = request.get("instruct_text", "用温柔甜美的女声说")
    
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    
    try:
        # 调用 TTS
        result = await mouth_adapter.process(text, instruct_text)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        # 转换为 WAV 格式返回
        import io
        import wave
        import numpy as np
        
        audio_array = np.array(result["audio"], dtype=np.float32)
        sample_rate = result["sample_rate"]
        
        # 转换为 16-bit PCM
        audio_int16 = (audio_array * 32767).astype(np.int16)
        
        # 创建 WAV 文件
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        wav_buffer.seek(0)
        
        from fastapi.responses import StreamingResponse
        return StreamingResponse(
            wav_buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"语音合成失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== 对话日志 API ==============

@app.get("/logs/dates")
async def list_log_dates():
    """列出所有有日志的日期"""
    from pathlib import Path
    log_dir = Path("/workspace/project-trinity/project-trinity/logs/conversations")
    
    if not log_dir.exists():
        return {"dates": []}
    
    dates = [f.stem for f in log_dir.glob("*.jsonl")]
    return {"dates": sorted(dates, reverse=True)}


@app.get("/logs/{date}")
async def get_logs_by_date(date: str):
    """
    获取指定日期的对话记录
    
    Args:
        date: 日期 YYYY-MM-DD
    """
    import json
    from pathlib import Path
    
    log_file = Path(f"/workspace/project-trinity/project-trinity/logs/conversations/{date}.jsonl")
    
    if not log_file.exists():
        return {"date": date, "conversations": [], "count": 0}
    
    conversations = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                conversations.append(json.loads(line))
    
    return {
        "date": date,
        "conversations": conversations,
        "count": len(conversations)
    }


@app.get("/logs/today")
async def get_today_logs():
    """获取今天的对话记录"""
    today = datetime.now().strftime("%Y-%m-%d")
    return await get_logs_by_date(today)


# ============== 主入口 ==============
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=settings.server.debug
    )
