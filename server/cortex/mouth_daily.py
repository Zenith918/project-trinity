"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  👄 CORTEX-MOUTH-DAILY (端口 9003)                                            ║
║  VoxCPM 1.5 - 极致低延迟配置                                                  ║
║                                                                              ║
║  注意: optimize=False 以支持流式输出                                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import io
import wave
import numpy as np
import torch
from loguru import logger
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, Response
from contextlib import asynccontextmanager
from typing import Optional, Generator
import time

# 全局实例
mouth = None


class DailyMouthHandler:
    """VoxCPM 1.5 处理器"""
    
    def __init__(self):
        self.model = None
        self.is_ready = False
        self.sample_rate = 24000
        
        # 配置
        self.config = {
            "steps": 2,
            "cfg_value": 1.0,
        }
        
    async def initialize(self):
        logger.info("=" * 60)
        logger.info("正在初始化 VoxCPM 1.5...")
        logger.info("=" * 60)
        
        try:
            from voxcpm import VoxCPM
            
            # 加载模型 - 禁用 optimize 以支持流式
            self.model = VoxCPM.from_pretrained(
                hf_model_id="openbmb/VoxCPM1.5",
                load_denoiser=False,
                optimize=False,  # 关键：禁用以支持流式
            )
            
            # 预热
            logger.info("预热推理...")
            _ = self.model.generate(
                text="预热测试",
                inference_timesteps=self.config["steps"],
                cfg_value=self.config["cfg_value"],
            )
            
            self.is_ready = True
            logger.success("✅ VoxCPM 1.5 初始化完成！")
            return True
            
        except Exception as e:
            logger.error(f"VoxCPM 初始化失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def synthesize(self, text: str, inference_timesteps: int = None, cfg_value: float = None) -> bytes:
        """一次性合成"""
        if not self.is_ready:
            return b""
        
        steps = inference_timesteps or self.config["steps"]
        cfg = cfg_value or self.config["cfg_value"]
            
        try:
            start_time = time.time()
            
            audio = self.model.generate(
                text=text,
                cfg_value=cfg,
                inference_timesteps=steps,
            )
            
            elapsed = (time.time() - start_time) * 1000
            logger.info(f"生成完成: {len(text)}字, {elapsed:.0f}ms")
            
            # 转换为 WAV
            audio_int16 = (audio * 32767).astype(np.int16)
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_int16.tobytes())
            
            return buffer.getvalue()
            
        except Exception as e:
            logger.error(f"推理失败: {e}")
            return b""

    def synthesize_stream(self, text: str, inference_timesteps: int = None, cfg_value: float = None):
        """流式合成"""
        if not self.is_ready:
            yield b""
            return
        
        steps = inference_timesteps or self.config["steps"]
        cfg = cfg_value or self.config["cfg_value"]
            
        try:
            start_time = time.time()
            first_chunk = True
            
            for chunk in self.model.generate_streaming(
                text=text,
                cfg_value=cfg,
                inference_timesteps=steps,
            ):
                if first_chunk:
                    ttfa = (time.time() - start_time) * 1000
                    logger.info(f"TTFA: {ttfa:.0f}ms")
                    first_chunk = False
                
                chunk_int16 = (chunk * 32767).astype(np.int16)
                yield chunk_int16.tobytes()
                
        except Exception as e:
            logger.error(f"流式推理失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            yield b""


@asynccontextmanager
async def lifespan(app: FastAPI):
    global mouth
    logger.info("👄 Cortex-Mouth-Daily 启动中...")
    
    mouth = DailyMouthHandler()
    await mouth.initialize()
    
    if mouth.is_ready:
        logger.success("✅ Mouth-Daily 就绪 (端口 9003)")
    
    yield
    logger.info("🛑 Mouth-Daily 关闭中...")


app = FastAPI(lifespan=lifespan, title="Cortex-Mouth-Daily")


@app.get("/health")
async def health():
    return {
        "service": "mouth-daily",
        "status": "ok" if mouth and mouth.is_ready else "loading",
        "model": "VoxCPM 1.5",
        "sample_rate": 24000,
        "config": mouth.config if mouth else {}
    }


@app.post("/tts")
async def tts(request: dict):
    if not mouth or not mouth.is_ready:
        return {"error": "Mouth not ready"}
    
    text = request.get("text", "")
    if not text:
        return {"error": "text is required"}
    
    inference_timesteps = request.get("inference_timesteps")
    cfg_value = request.get("cfg_value")
    
    audio_bytes = mouth.synthesize(text, inference_timesteps, cfg_value)
    
    if not audio_bytes:
        return {"error": "Synthesis failed"}
    
    return Response(content=audio_bytes, media_type="audio/wav")


@app.post("/tts/stream")
async def tts_stream(request: dict):
    if not mouth or not mouth.is_ready:
        return {"error": "Mouth not ready"}
    
    text = request.get("text", "")
    if not text:
        return {"error": "text is required"}
    
    inference_timesteps = request.get("inference_timesteps")
    cfg_value = request.get("cfg_value")
    
    return StreamingResponse(
        mouth.synthesize_stream(text, inference_timesteps, cfg_value),
        media_type="audio/pcm",
        headers={"X-Sample-Rate": "24000"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9003)
