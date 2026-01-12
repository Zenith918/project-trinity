"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  👂 CORTEX-EAR SERVER (端口 9002)                                             ║
║  只负责 ASR 语音识别 (FunASR/SenseVoice)                                      ║
║  可独立重启，不影响 Brain 和 Mouth                                            ║
║                                                                              ║
║  💡 开发提示: 此服务极轻量，重启只需 ~5s                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import io
import wave
import asyncio
import subprocess
import numpy as np
from loguru import logger

# ═══════════════════════════════════════════════════════════════════════════════
# 🔧 自动端口清理
# ═══════════════════════════════════════════════════════════════════════════════
SERVICE_PORT = 9002

def kill_port(port: int):
    """杀掉占用指定端口的进程"""
    try:
        subprocess.run(f"fuser -k {port}/tcp 2>/dev/null || true", shell=True, timeout=5)
    except Exception:
        pass

kill_port(SERVICE_PORT)
from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
from typing import Tuple

# 全局模型实例
ear = None

class EarHandler:
    """FunASR (SenseVoice) 语音识别处理器"""
    
    def __init__(self, model_name: str = "iic/SenseVoiceSmall", device: str = "cuda:0"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.is_ready = False
        self._lock = asyncio.Lock()
    
    async def initialize(self):
        logger.info(f"正在初始化 FunASR: {self.model_name}")
        try:
            from funasr import AutoModel
            self.model = AutoModel(
                model=self.model_name,
                trust_remote_code=True,
                device=self.device
            )
            self.is_ready = True
            logger.success(f"FunASR 初始化成功")
            return True
        except Exception as e:
            logger.error(f"FunASR 初始化失败: {e}")
            return False
    
    async def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> dict:
        """语音转文字"""
        if not self.is_ready:
            raise RuntimeError("EarHandler 未初始化")
        
        async with self._lock:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._inference, audio_data, sample_rate)
            return result
    
    def _inference(self, audio_data: np.ndarray, sample_rate: int) -> dict:
        result = self.model.generate(
            input=audio_data,
            cache={},
            language="auto",
            use_itn=True,
            batch_size_s=60
        )
        
        text = result[0]["text"] if result else ""
        emotion, clean_text = self._parse_emotion(text)
        
        return {
            "text": clean_text,
            "raw_text": text,
            "emotion": emotion,
            "language": "zh" if any('\u4e00' <= c <= '\u9fff' for c in clean_text) else "en"
        }
    
    def _parse_emotion(self, text: str) -> Tuple[str, str]:
        emotion_map = {
            "HAPPY": "happy", "SAD": "sad", "ANGRY": "angry",
            "FEARFUL": "fearful", "DISGUSTED": "disgusted",
            "SURPRISED": "surprised", "NEUTRAL": "neutral"
        }
        emotion = "neutral"
        clean_text = text
        for tag, name in emotion_map.items():
            if f"<|{tag}|>" in text:
                emotion = name
                clean_text = text.replace(f"<|{tag}|>", "").strip()
                break
        # 清理其他标签
        for tag in ["<|zh|>", "<|en|>", "<|EMO_UNKNOWN|>", "<|Speech|>", "<|withitn|>"]:
            clean_text = clean_text.replace(tag, "")
        return emotion, clean_text.strip()
    
    async def shutdown(self):
        if self.model:
            del self.model
            self.model = None
        self.is_ready = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ear
    logger.info("👂 Cortex-Ear Server 启动中...")
    
    ear = EarHandler()
    await ear.initialize()
    
    logger.success("✅ Ear Server 就绪 (端口 9002)")
    yield
    
    logger.info("🛑 Ear Server 关闭中...")
    if ear:
        await ear.shutdown()

app = FastAPI(lifespan=lifespan, title="Cortex-Ear")

@app.get("/health")
async def health():
    return {
        "service": "ear",
        "status": "ok" if ear and ear.is_ready else "loading",
        "model": "SenseVoiceSmall"
    }

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """
    语音转文字
    
    接受: WAV/PCM 音频文件
    返回: {"text": "识别结果", "emotion": "情感", "language": "语言"}
    """
    if not ear or not ear.is_ready:
        raise HTTPException(status_code=503, detail="Ear not ready")
    
    try:
        # 读取音频
        audio_bytes = await file.read()
        
        # 解析 WAV
        with io.BytesIO(audio_bytes) as wav_io:
            with wave.open(wav_io, 'rb') as wf:
                sample_rate = wf.getframerate()
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                frames = wf.readframes(wf.getnframes())
        
        # 转换为 numpy
        if sampwidth == 2:
            audio_array = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            audio_array = np.frombuffer(frames, dtype=np.float32)
        
        # 单声道
        if n_channels > 1:
            audio_array = audio_array.reshape(-1, n_channels).mean(axis=1)
        
        # 识别
        result = await ear.transcribe(audio_array, sample_rate)
        return result
        
    except Exception as e:
        logger.error(f"语音识别失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    kill_port(SERVICE_PORT)
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT)





