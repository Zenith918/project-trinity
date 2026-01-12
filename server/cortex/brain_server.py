"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  🧠 CORTEX-BRAIN SERVER (端口 9000)                                           ║
║  只负责 LLM 推理 (Qwen2.5-VL)                                                 ║
║  可独立重启，不影响 Mouth 和 Ear                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import subprocess
from loguru import logger

# ═══════════════════════════════════════════════════════════════════════════════
# 🔧 自动端口清理
# ═══════════════════════════════════════════════════════════════════════════════
SERVICE_PORT = 9000

def kill_port(port: int):
    """杀掉占用指定端口的进程"""
    try:
        subprocess.run(f"fuser -k {port}/tcp 2>/dev/null || true", shell=True, timeout=5)
    except Exception:
        pass

kill_port(SERVICE_PORT)
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager

# 全局模型实例
brain = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global brain
    logger.info("🧠 Cortex-Brain Server 启动中...")
    
    from server.cortex.models.brain import BrainHandler
    brain = BrainHandler()
    await brain.initialize()
    
    logger.success("✅ Brain Server 就绪 (端口 9000)")
    yield
    
    logger.info("🛑 Brain Server 关闭中...")
    if brain:
        await brain.shutdown()

app = FastAPI(lifespan=lifespan, title="Cortex-Brain")

@app.get("/health")
async def health():
    return {
        "service": "brain",
        "status": "ok" if brain and brain.is_ready else "loading",
        "model": "Qwen2.5-VL-7B-AWQ"
    }

@app.post("/chat")
async def chat(request: dict):
    """非流式聊天"""
    if not brain or not brain.is_ready:
        return {"error": "Brain not ready"}
    return await brain.generate(request)

@app.post("/chat/stream")
async def chat_stream(request: dict):
    """流式聊天 (SSE)"""
    if not brain or not brain.is_ready:
        return {"error": "Brain not ready"}
    
    async def event_generator():
        try:
            async for token in brain.generate_stream(request):
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

if __name__ == "__main__":
    import uvicorn
    kill_port(SERVICE_PORT)  # 启动前再次确保端口清理
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT)

