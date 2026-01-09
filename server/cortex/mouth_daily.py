"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  👄 CORTEX-MOUTH-DAILY (端口 9003)                                            ║
║  VoxCPM 1.5 - 极致低延迟配置                                                  ║
║                                                                              ║
║  🔥 optimize=True + 禁用 CUDA Graph = TTFA ~285ms (比 optimize=False 快 37%)   ║
║  💡 首次流式调用会触发 JIT 编译 (~13秒)，之后稳定在 ~285ms                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
# 🔑 关键：在导入 torch 之前禁用 CUDA Graph
os.environ['TORCHINDUCTOR_CUDAGRAPHS'] = '0'

import torch
# 双重保险：通过 config 禁用
torch._inductor.config.triton.cudagraphs = False

import io
import wave
import numpy as np
from loguru import logger
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, Response
from contextlib import asynccontextmanager
import time

mouth = None


class DailyMouthHandler:
    """VoxCPM 1.5 处理器 - optimize=True + 禁用 CUDA Graph"""
    
    def __init__(self):
        self.model = None
        self.is_ready = False
        self.sample_rate = 24000
        self.config = {"steps": 2, "cfg_value": 1.0}
        
    async def initialize(self):
        logger.info("=" * 60)
        logger.info("正在初始化 VoxCPM 1.5 (optimize=True, cudagraphs=False)...")
        logger.info("=" * 60)
        
        try:
            from voxcpm import VoxCPM
            
            # 🔥 启用 torch.compile 优化
            self.model = VoxCPM.from_pretrained(
                hf_model_id="openbmb/VoxCPM1.5",
                load_denoiser=False,
                optimize=True,
            )
            
            # 预热 1: 非流式
            logger.info("预热 1/2: 非流式推理...")
            _ = self.model.generate(
                text="预热",
                inference_timesteps=self.config["steps"],
                cfg_value=self.config["cfg_value"],
            )
            
            # 预热 2: 流式 (触发完整 JIT)
            logger.info("预热 2/2: 流式推理 (触发 JIT 编译, 约 13 秒)...")
            for chunk in self.model.generate_streaming(
                text="流式预热",
                inference_timesteps=self.config["steps"],
                cfg_value=self.config["cfg_value"],
            ):
                pass
            
            self.is_ready = True
            logger.success("✅ VoxCPM 1.5 初始化完成 (TTFA ~285ms)")
            return True
            
        except Exception as e:
            logger.error(f"VoxCPM 初始化失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def synthesize(self, text, inference_timesteps=None, cfg_value=None):
        if not self.is_ready:
            return b""
        steps = inference_timesteps or self.config["steps"]
        cfg = cfg_value or self.config["cfg_value"]
        try:
            start = time.time()
            audio = self.model.generate(text=text, cfg_value=cfg, inference_timesteps=steps)
            logger.info(f"生成: {len(text)}字, {(time.time()-start)*1000:.0f}ms")
            audio_int16 = (audio * 32767).astype(np.int16)
            buf = io.BytesIO()
            with wave.open(buf, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_int16.tobytes())
            return buf.getvalue()
        except Exception as e:
            logger.error(f"推理失败: {e}")
            return b""

    def synthesize_stream(self, text, inference_timesteps=None, cfg_value=None):
        if not self.is_ready:
            yield b""
            return
        steps = inference_timesteps or self.config["steps"]
        cfg = cfg_value or self.config["cfg_value"]
        try:
            start = time.time()
            first = True
            for chunk in self.model.generate_streaming(text=text, cfg_value=cfg, inference_timesteps=steps):
                if first:
                    logger.info(f"TTFA: {(time.time()-start)*1000:.0f}ms")
                    first = False
                chunk_int16 = (chunk * 32767).astype(np.int16)
                yield chunk_int16.tobytes()
        except Exception as e:
            logger.error(f"流式失败: {e}")
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
    logger.info("🛑 Mouth-Daily 关闭")


app = FastAPI(lifespan=lifespan, title="Cortex-Mouth-Daily")


@app.get("/health")
async def health():
    return {
        "service": "mouth-daily",
        "status": "ok" if mouth and mouth.is_ready else "loading",
        "model": "VoxCPM 1.5 (optimized, cudagraph=off)",
        "sample_rate": 24000,
        "ttfa_target": "~285ms",
        "config": mouth.config if mouth else {}
    }


@app.post("/tts")
async def tts(request: dict):
    if not mouth or not mouth.is_ready:
        return {"error": "Not ready"}
    text = request.get("text", "")
    if not text:
        return {"error": "text required"}
    audio = mouth.synthesize(text, request.get("inference_timesteps"), request.get("cfg_value"))
    if not audio:
        return {"error": "failed"}
    return Response(content=audio, media_type="audio/wav")


@app.post("/tts/stream")
async def tts_stream(request: dict):
    if not mouth or not mouth.is_ready:
        return {"error": "Not ready"}
    text = request.get("text", "")
    if not text:
        return {"error": "text required"}
    return StreamingResponse(
        mouth.synthesize_stream(text, request.get("inference_timesteps"), request.get("cfg_value")),
        media_type="audio/pcm",
        headers={"X-Sample-Rate": "24000"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9003)
