import os
import sys
import socket
import psutil
from loguru import logger
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import json

# ==========================================
# 0. 端口抢占与清理 (Port Guard)
# ==========================================
def ensure_port_available(port: int):
    """确保端口可用，如果被占用则杀掉占用进程"""
    logger.info(f"🛡️ 检查端口 {port}...")
    try:
        # 尝试绑定端口
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('0.0.0.0', port))
        sock.close()
        logger.success(f"端口 {port} 可用")
        return
    except OSError:
        logger.warning(f"端口 {port} 被占用，正在寻找占用者...")
    
    # 查找并杀掉占用进程
    killed = False
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            for conn in proc.connections(kind='inet'):
                if conn.laddr.port == port:
                    logger.warning(f"发现占用进程: PID={proc.info['pid']} Name={proc.info['name']}")
                    proc.kill()
                    killed = True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
            
    if killed:
        logger.success(f"已清理占用端口 {port} 的进程")
    else:
        logger.error(f"无法清理端口 {port}，可能权限不足或非 Python 进程占用")

# 在导入大模型前执行检查
ensure_port_available(9000)

# 强制添加 CosyVoice 路径
COSYVOICE_PATH = "/workspace/CosyVoice"
if os.path.exists(COSYVOICE_PATH) and COSYVOICE_PATH not in sys.path:
    sys.path.insert(0, COSYVOICE_PATH)

from .models.brain import BrainHandler
from .models.mouth import MouthHandler

# 全局模型实例
brain = None
mouth = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global brain, mouth
    logger.info("🧠 Cortex Model Server 正在启动...")
    
    # 1. 加载 Brain (Qwen2.5-VL)
    # 显存占用最大，优先加载
    try:
        brain = BrainHandler()
        await brain.initialize()
    except Exception as e:
        logger.error(f"Brain 加载失败: {e}")
        
    # 2. 加载 Mouth (CosyVoice 3.0)
    try:
        mouth = MouthHandler()
        await mouth.initialize()
    except Exception as e:
        logger.error(f"Mouth 加载失败: {e}")
        
    logger.success("✅ Cortex Server 就绪")
    yield
    
    # 清理
    logger.info("🛑 Cortex Server 关闭中...")
    if brain: await brain.shutdown()
    if mouth: await mouth.shutdown()

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health_check():
    status = {
        "brain": brain.is_ready if brain else False,
        "mouth": mouth.is_ready if mouth else False
    }
    return {"status": "ok", "modules": status}

@app.post("/brain/chat")
async def chat(request: dict):
    """非流式聊天 (兼容旧接口)"""
    if not brain or not brain.is_ready:
        return {"error": "Brain not ready"}
    return await brain.generate(request)

@app.post("/brain/chat/stream")
async def chat_stream(request: dict):
    """
    真正的流式聊天 - SSE (Server-Sent Events)
    每个 token 生成后立即发送，TTFT 目标 <200ms
    """
    if not brain or not brain.is_ready:
        return {"error": "Brain not ready"}
    
    async def event_generator():
        try:
            async for token in brain.generate_stream(request):
                # SSE 格式: data: {json}\n\n
                yield f"data: {json.dumps({'token': token})}\n\n"
            # 发送结束标记
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # 禁用 nginx 缓冲
        }
    )

@app.post("/mouth/tts")
async def tts(request: dict):
    if not mouth or not mouth.is_ready:
        return {"error": "Mouth not ready"}
    return await mouth.synthesize(request)
