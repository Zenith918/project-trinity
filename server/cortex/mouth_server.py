"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  👄 CORTEX-MOUTH SERVER (端口 9001) - CosyVoice 3.0                           ║                                           ║
║  可独立重启，不影响 Brain 和 Ear (~60s 加载时间)                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║        ║
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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import asyncio
import json

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

@app.websocket("/tts/ws")
async def tts_websocket(websocket: WebSocket):
    """
    🆕 双向流式 TTS 接口 - 阿里级边进边出架构
    
    协议:
    1. Client 发送文本片段 (Text Frames)，可以逐字发送
    2. Server 实时返回音频片段 (Binary Frames)
    3. Client 发送空文本 ("") 表示输入结束
    
    特性:
    - 动态触发阈值: 5字首包 / 12字语感包 / 强标点触发
    - 零拷贝传输: send_bytes 直接推送二进制
    - 非阻塞队列: 接收和推理在独立任务
    """
    await websocket.accept()
    logger.info("🔌 WebSocket 连接建立")
    
    if not mouth or not mouth.is_ready:
        await websocket.close(code=1011, reason="Mouth not ready")
        return

    # 创建一个 asyncio Queue 作为文本缓冲区
    text_queue = asyncio.Queue()
    input_ended = False
    
    async def receive_text_loop():
        """接收前端发来的文本流 (独立异步任务)"""
        nonlocal input_ended
        try:
            while True:
                data = await websocket.receive_text()
                if data:
                    # 逐字放入队列，让 synthesize_stream 可以边进边出
                    for char in data:
                        await text_queue.put(char)
                else:
                    # 空消息表示输入结束
                    input_ended = True
                    break
        except WebSocketDisconnect:
            input_ended = True
        except Exception as e:
            logger.error(f"WebSocket Receive Error: {e}")
            input_ended = True
        finally:
            # 发送结束标记
            await text_queue.put(None)

    async def text_iterator():
        """将 Queue 转换为 AsyncIterator[str] 供 mouth 使用"""
        while True:
            char = await text_queue.get()
            if char is None:
                break
            yield char
    
    # 启动接收任务 (非阻塞)
    receive_task = asyncio.create_task(receive_text_loop())
    
    try:
        # 🚀 启动合成并发送音频
        # synthesize_stream 已支持 AsyncIterator[str]，实现边进边出
        async for audio_chunk in mouth.synthesize_stream(text_iterator()):
            if audio_chunk:  # 过滤空块
                await websocket.send_bytes(audio_chunk)
            
    except Exception as e:
        logger.error(f"WebSocket TTS Error: {e}")
    finally:
        receive_task.cancel()
        try:
            await websocket.close()
            logger.info("🔌 WebSocket 连接关闭")
        except:
            pass

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
    文本转语音 - 支持流式和非流式模式
    
    请求体:
    {
        "text": "要合成的文本",
        "instruct_text": "用温柔甜美的女声说",  // 暂未使用
        "stream": false  // true=流式返回 (推荐)
    }
    
    🆕 改进:
    - stream=true 时使用动态阈值架构，享受更低延迟
    - stream=false 时等待完整音频后返回
    """
    if not mouth or not mouth.is_ready:
        return {"error": "Mouth not ready"}
    
    text = request.get("text", "")
    instruct_text = request.get("instruct_text", "用温柔甜美的女声说")
    stream = request.get("stream", False)
    
    if not text:
        return {"error": "text is required"}
    
    if stream:
        # 🆕 流式模式：直接传入 str，synthesize_stream 内部会归一化处理
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
    🆕 流式 TTS - 阿里级 200ms TTFA 架构
    
    特性:
    - 动态触发阈值: 5字首包 / 12字语感包 / 强标点触发
    - 边生成边发送音频块
    - 返回格式: chunked WAV data (首包带头，后续 PCM)
    """
    if not mouth or not mouth.is_ready:
        return {"error": "Mouth not ready"}
    
    text = request.get("text", "")
    instruct_text = request.get("instruct_text", "用温柔甜美的女声说")
    
    if not text:
        return {"error": "text is required"}
    
    # 🆕 直接传入 str，synthesize_stream 内部归一化处理
    return StreamingResponse(
        mouth.synthesize_stream(text, instruct_text),
        media_type="audio/wav",
        headers={
            "X-Streaming": "true",
            "Cache-Control": "no-cache",
            "X-TTFA-Target": "200ms"
        }
    )

# ═══════════════════════════════════════════════════════════════════════════════
# 🎛️ 动态配置 API
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/config")
async def get_config():
    """获取当前配置"""
    from trinity_config import config
    return {
        "config": config.to_dict(),
        "description": {
            "n_timesteps": "Flow ODE 步数 (2=极速有电磁音, 5=平衡, 10=高质量)",
            "token_hop_len": "LLM token 缓冲 (5=极速可能卡顿, 10=平衡, 25=高质量)",
            "first_chunk_threshold": "首包触发字符数",
            "normal_chunk_threshold": "后续触发字符数"
        }
    }

@app.post("/config")
async def update_config(request: dict):
    """
    动态更新配置 (无需重启服务)
    
    示例请求:
    {
        "n_timesteps": 5,
        "token_hop_len": 10
    }
    """
    from trinity_config import config
    
    # 更新配置
    updated = config.update(**request)
    
    # 同步更新模型的 token_hop_len (如果已加载)
    if mouth and mouth.model and "token_hop_len" in updated:
        mouth.model.model.token_hop_len = config.token_hop_len
        logger.info(f"🔧 已同步更新 model.token_hop_len = {config.token_hop_len}")
    
    return {
        "status": "updated",
        "changes": updated,
        "current": config.to_dict()
    }

if __name__ == "__main__":
    import uvicorn
    kill_port(SERVICE_PORT)
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT)

