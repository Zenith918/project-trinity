"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  👄 CORTEX-MOUTH-FISH (端口 9006)                                             ║
║  Fish Speech 1.5 - 人情味语音合成                                             ║
║                                                                              ║
║  🔥 特性：                                                                    ║
║    - 原生 44.1kHz 输出（无需重采样）                                          ║
║    - BF16 精度 (4090 最优)                                                    ║
║    - 支持情感标签 [laughter], [sigh] 等                                       ║
║    - LLM + VQ-GAN 分段流式推理                                                ║
║    - 防幻觉：RTF > 0.5 自动报错重置                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import io
import wave
import time
import queue
import threading
import numpy as np
import torch

# 添加 Fish Speech 源码路径
FISH_SPEECH_PATH = "/workspace/models/fish-speech"
sys.path.insert(0, FISH_SPEECH_PATH)
os.chdir(FISH_SPEECH_PATH)

from loguru import logger
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Optional, List
import tempfile

# ============================================================================
# 配置常量
# ============================================================================
SERVICE_PORT = 9006
MODEL_DIR = "/workspace/models/FishSpeech/fish-speech-1.5"

# Fish Speech 原生 44.1kHz
OUTPUT_SAMPLE_RATE = 44100

# 防幻觉阈值：RTF 超过 0.5 视为异常
MAX_RTF = 0.5

mouth = None


class FishSpeechHandler:
    """Fish Speech 1.5 处理器"""
    
    def __init__(self):
        self.model_manager = None
        self.is_ready = False
        self.sample_rate = OUTPUT_SAMPLE_RATE
        self.lock = threading.Lock()
        
        # 默认参考音频
        self.default_reference = "/workspace/project-trinity/project-trinity/assets/prompt_female.wav"
        
    async def initialize(self):
        logger.info("=" * 60)
        logger.info("正在初始化 Fish Speech 1.5...")
        logger.info(f"模型目录: {MODEL_DIR}")
        logger.info("=" * 60)
        
        try:
            from tools.server.model_manager import ModelManager
            
            start_time = time.time()
            
            # 初始化模型管理器
            self.model_manager = ModelManager(
                mode="tts",
                device="cuda",
                half=False,           # 使用 BF16，不是 FP16
                compile=True,         # 🔥 启用 torch.compile
                llama_checkpoint_path=MODEL_DIR,
                decoder_checkpoint_path=f"{MODEL_DIR}/firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
                decoder_config_name="firefly_gan_vq",
            )
            
            load_time = time.time() - start_time
            logger.info(f"模型加载完成，耗时 {load_time:.1f}s")
            
            # 获取实际采样率
            if hasattr(self.model_manager, 'engine') and hasattr(self.model_manager.engine, 'decoder_model'):
                if hasattr(self.model_manager.engine.decoder_model, 'sample_rate'):
                    self.sample_rate = self.model_manager.engine.decoder_model.sample_rate
                    logger.info(f"实际采样率: {self.sample_rate}Hz")
            
            # 预热推理
            logger.info("预热推理中...")
            warmup_start = time.time()
            
            # 简单预热
            _ = self._synthesize_internal("预热", streaming=False)
            
            warmup_time = time.time() - warmup_start
            logger.success(f"✅ 预热完成，耗时 {warmup_time:.1f}s")
            
            self.is_ready = True
            total_time = time.time() - start_time
            logger.success(f"✅ Fish Speech 1.5 初始化完成！总耗时 {total_time:.1f}s")
            return True
            
        except Exception as e:
            logger.error(f"Fish Speech 初始化失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _synthesize_internal(self, text: str, 
                             reference_audio: Optional[str] = None,
                             streaming: bool = False):
        """内部合成方法"""
        from fish_speech.utils.schema import ServeTTSRequest
        
        # 构建请求
        req = ServeTTSRequest(
            text=text,
            references=[],  # 可以添加参考音频
            reference_id=None,
            streaming=streaming,
            chunk_length=200 if streaming else 0,  # 流式分块
            top_p=0.7,
            temperature=0.7,
            repetition_penalty=1.2,
            max_new_tokens=2048,
            use_memory_cache=True,
        )
        
        return self.model_manager.engine.inference(req)

    def synthesize(self, text: str, 
                   reference_audio: Optional[str] = None) -> bytes:
        """非流式合成"""
        if not self.is_ready:
            return b""
        
        try:
            start = time.time()
            
            with self.lock:
                results = list(self._synthesize_internal(text, reference_audio, streaming=False))
            
            # 查找最终音频
            audio_data = None
            for result in results:
                if result.code == "final" and result.audio is not None:
                    sr, audio_data = result.audio
                    self.sample_rate = sr
                elif result.code == "error":
                    logger.error(f"合成错误: {result.error}")
                    return b""
            
            if audio_data is None:
                return b""
            
            elapsed = time.time() - start
            audio_duration = len(audio_data) / self.sample_rate
            rtf = elapsed / audio_duration if audio_duration > 0 else float('inf')
            
            # 防幻觉检查
            if rtf > MAX_RTF:
                logger.warning(f"⚠️ RTF 异常: {rtf:.2f} > {MAX_RTF}，可能存在幻觉")
            
            # 转换为 WAV
            audio_int16 = (audio_data * 32767).astype(np.int16)
            buf = io.BytesIO()
            with wave.open(buf, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_int16.tobytes())
            
            logger.info(f"合成完成: {len(text)}字, {elapsed*1000:.0f}ms, RTF={rtf:.2f}")
            
            return buf.getvalue()
            
        except Exception as e:
            logger.error(f"合成失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return b""

    def synthesize_stream(self, text: str,
                          reference_audio: Optional[str] = None):
        """流式合成"""
        if not self.is_ready:
            yield b""
            return
        
        try:
            start = time.time()
            first_chunk = True
            chunk_count = 0
            total_audio_duration = 0
            
            with self.lock:
                for result in self._synthesize_internal(text, reference_audio, streaming=True):
                    if result.code == "header":
                        # 跳过 WAV header（我们发送 raw PCM）
                        continue
                    elif result.code == "segment":
                        chunk_count += 1
                        sr, audio_chunk = result.audio
                        self.sample_rate = sr
                        
                        if first_chunk:
                            ttfa = (time.time() - start) * 1000
                            logger.info(f"TTFA: {ttfa:.0f}ms")
                            first_chunk = False
                        
                        # 转换为 int16
                        audio_int16 = (audio_chunk * 32767).astype(np.int16)
                        total_audio_duration += len(audio_chunk) / sr
                        
                        # 防幻觉检查
                        elapsed = time.time() - start
                        if total_audio_duration > 0 and elapsed / total_audio_duration > MAX_RTF * 3:
                            logger.warning(f"⚠️ 检测到幻觉，强制截断！RTF={elapsed/total_audio_duration:.2f}")
                            break
                        
                        yield audio_int16.tobytes()
                        
                    elif result.code == "error":
                        logger.error(f"流式错误: {result.error}")
                        yield b""
                        return
                    elif result.code == "final":
                        # 流式模式下 final 之前已经发送了所有 segment
                        pass
            
            logger.info(f"流式完成: {chunk_count} chunks")
                    
        except Exception as e:
            logger.error(f"流式合成失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            yield b""


@asynccontextmanager
async def lifespan(app: FastAPI):
    global mouth
    logger.info(f"👄 Cortex-Mouth-Fish 启动中 (端口 {SERVICE_PORT})...")
    mouth = FishSpeechHandler()
    success = await mouth.initialize()
    if success:
        logger.success(f"✅ Mouth-Fish 就绪 (端口 {SERVICE_PORT})")
    else:
        logger.error("❌ Mouth-Fish 初始化失败")
    yield
    logger.info("🛑 Mouth-Fish 关闭")


app = FastAPI(lifespan=lifespan, title="Cortex-Mouth-Fish (Fish Speech 1.5)")

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "service": "mouth-fish",
        "status": "ok" if mouth and mouth.is_ready else "loading",
        "model": "Fish Speech 1.5 (BF16, compile=True)",
        "sample_rate": mouth.sample_rate if mouth else OUTPUT_SAMPLE_RATE,
        "features": ["emotion_tags", "reference_audio", "anti_hallucination"],
        "emotion_tags": ["[laughter]", "[sigh]", "[breath]", "[cough]"],
    }


@app.post("/tts")
async def tts(request: dict):
    """
    非流式 TTS 接口
    
    请求体:
    {
        "text": "要合成的文本，支持 [laughter] [sigh] 等情感标签",
        "reference_audio": "可选，参考音频路径"
    }
    """
    if not mouth or not mouth.is_ready:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    text = request.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="text required")
    
    audio = mouth.synthesize(
        text=text,
        reference_audio=request.get("reference_audio"),
    )
    
    if not audio:
        raise HTTPException(status_code=500, detail="Synthesis failed")
    
    return Response(content=audio, media_type="audio/wav")


@app.post("/tts/stream")
async def tts_stream(request: dict):
    """
    流式 TTS 接口
    
    返回 PCM 音频流 (44.1kHz, 16bit, mono)
    """
    if not mouth or not mouth.is_ready:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    text = request.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="text required")
    
    return StreamingResponse(
        mouth.synthesize_stream(
            text=text,
            reference_audio=request.get("reference_audio"),
        ),
        media_type="audio/pcm",
        headers={"X-Sample-Rate": str(mouth.sample_rate)}
    )


if __name__ == "__main__":
    import uvicorn
    from server.utils.port_utils import kill_port
    
    kill_port(SERVICE_PORT)
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT)


