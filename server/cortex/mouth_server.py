"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  👄 CORTEX-MOUTH SERVER (端口 9001) - CosyVoice 3.0                           ║                                           ║
║  可独立重启，不影响 Brain 和 Ear (~60s 加载时间)                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ⚠️ 注意: CosyVoice TTFT ~10秒       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import io
import wave
import subprocess
import numpy as np
from loguru import logger

# ═══════════════════════════════════════════════════════════════════════════════
# 🔧 自动端口清理
# ═══════════════════════════════════════════════════════════════════════════════
SERVICE_PORT = 9001

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

# 添加 CosyVoice 路径
COSYVOICE_PATH = "/workspace/CosyVoice"
MATCHA_PATH = "/workspace/CosyVoice/third_party/Matcha-TTS"
if os.path.exists(COSYVOICE_PATH) and COSYVOICE_PATH not in sys.path:
    sys.path.insert(0, COSYVOICE_PATH)
if os.path.exists(MATCHA_PATH) and MATCHA_PATH not in sys.path:
    sys.path.insert(0, MATCHA_PATH)

# 全局模型实例
mouth = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global mouth
    logger.info("👄 Cortex-Mouth Server 启动中...")
    
    from server.cortex.models.mouth import MouthHandler
    mouth = MouthHandler()
    await mouth.initialize()
    
    logger.success("✅ Mouth Server 就绪 (端口 9001)")
    yield
    
    logger.info("🛑 Mouth Server 关闭中...")
    if mouth:
        await mouth.shutdown()

app = FastAPI(lifespan=lifespan, title="Cortex-Mouth")

@app.get("/health")
async def health():
    return {
        "service": "mouth",
        "status": "ok" if mouth and mouth.is_ready else "loading",
        "model": "CosyVoice3-0.5B"
    }

@app.post("/tts")
async def tts(request: dict):
    """
    文本转语音
    
    请求体:
    {
        "text": "要合成的文本",
        "instruct_text": "用温柔甜美的女声说",
        "stream": false  // 是否流式返回
    }
    """
    if not mouth or not mouth.is_ready:
        return {"error": "Mouth not ready"}
    
    text = request.get("text", "")
    instruct_text = request.get("instruct_text", "用温柔甜美的女声说")
    stream = request.get("stream", False)
    
    if not text:
        return {"error": "text is required"}
    
    if stream:
        # 流式返回音频块
        return StreamingResponse(
            mouth.synthesize_stream(text, instruct_text),
            media_type="audio/wav",
            headers={"X-Streaming": "true"}
        )
    else:
        # 非流式: 等待完整音频
        result = await mouth.synthesize({"text": text, "instruct_text": instruct_text})
        
        if "error" in result:
            return result
        
        return StreamingResponse(
            io.BytesIO(result["audio_bytes"]),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"}
        )

@app.post("/tts/stream")
async def tts_stream(request: dict):
    """
    流式 TTS - 边生成边发送音频块
    返回格式: chunked WAV data
    """
    if not mouth or not mouth.is_ready:
        return {"error": "Mouth not ready"}
    
    text = request.get("text", "")
    instruct_text = request.get("instruct_text", "用温柔甜美的女声说")
    
    if not text:
        return {"error": "text is required"}
    
    async def audio_stream():
        async for chunk in mouth.synthesize_stream(text, instruct_text):
            yield chunk
    
    return StreamingResponse(
        audio_stream(),
        media_type="audio/wav",
        headers={
            "X-Streaming": "true",
            "Cache-Control": "no-cache"
        }
    )

if __name__ == "__main__":
    import uvicorn
    kill_port(SERVICE_PORT)
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT)

