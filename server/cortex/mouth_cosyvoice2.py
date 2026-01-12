#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CosyVoice2 0.5B FastAPI 服务
端口: 9005
环境: cosyvoice2_env (torch 2.3.1, flash-attn 2.5.8)

核心特性:
- 真流式输出 (stream=True)
- 44.1kHz 采样率 (从 22.05kHz 重采样)
- FP16 精度
- 预热机制确保 TTFA < 300ms
- RTF 实时监控
"""
import os
import sys
import time
import io
import wave

# ═══════════════════════════════════════════════════════════════════════════════
# 【铁律】日志配置 - 必须同时输出到文件和终端，确保可追踪！
# ═══════════════════════════════════════════════════════════════════════════════
from loguru import logger

# 移除默认 handler，重新配置
logger.remove()
# 终端输出 (彩色)
logger.add(sys.stderr, level="DEBUG", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")
# 文件输出 (详细)
LOG_FILE = "/tmp/cv2_new.log"
logger.add(LOG_FILE, level="DEBUG", format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}", rotation="10 MB")
logger.info(f"📝 日志文件: {LOG_FILE}")

# 添加 CosyVoice 到路径
sys.path.insert(0, '/workspace/models/CosyVoice')
os.chdir('/workspace/models/CosyVoice')

import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Optional
import tempfile

# ═══════════════════════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════════════════════
SERVICE_PORT = 9005
MODEL_DIR = "/workspace/models/CosyVoice/pretrained_models/iic/CosyVoice2-0___5B"
NATIVE_SAMPLE_RATE = 22050  # CosyVoice2 原生采样率
TARGET_SAMPLE_RATE = 44100  # 目标采样率

# ═══════════════════════════════════════════════════════════════════════════════
# 端口管理
# ═══════════════════════════════════════════════════════════════════════════════
def kill_port(port: int):
    """杀死占用指定端口的进程"""
    import subprocess
    try:
        result = subprocess.run(
            f"ss -tlnp | grep :{port} | awk '{{print $6}}' | grep -oP '(?<=pid=)\\d+' | xargs -r kill -9",
            shell=True, capture_output=True, text=True
        )
        if result.returncode == 0:
            logger.info(f"已清理端口 {port}")
    except Exception as e:
        logger.warning(f"清理端口失败: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# 请求模型
# ═══════════════════════════════════════════════════════════════════════════════
class TTSRequest(BaseModel):
    text: str
    speaker_id: Optional[str] = None
    speed: float = 1.0

# ═══════════════════════════════════════════════════════════════════════════════
# CosyVoice2 服务
# ═══════════════════════════════════════════════════════════════════════════════
class CosyVoice2Service:
    def __init__(self):
        self.model = None
        self.resampler = None
        self.ready = False
        self.default_speaker = None
        
    def initialize(self):
        """
        初始化 CosyVoice2 模型
        【重要】每一步都打印详细日志，方便排错！
        """
        logger.info("=" * 70)
        logger.info("🚀 开始初始化 CosyVoice2 0.5B")
        logger.info("=" * 70)
        
        try:
            # ═══════════════════════════════════════════════════════════════
            # Step 1: 导入模块
            # ═══════════════════════════════════════════════════════════════
            logger.info("[1/6] 导入 CosyVoice2 模块...")
            step_start = time.time()
            from cosyvoice.cli.cosyvoice import CosyVoice2
            logger.info(f"[1/6] ✅ 模块导入完成 ({time.time()-step_start:.1f}s)")
            
            # ═══════════════════════════════════════════════════════════════
            # Step 2: 检查模型目录
            # ═══════════════════════════════════════════════════════════════
            logger.info(f"[2/6] 检查模型目录: {MODEL_DIR}")
            if not os.path.exists(MODEL_DIR):
                raise FileNotFoundError(f"模型目录不存在: {MODEL_DIR}")
            
            files = os.listdir(MODEL_DIR)
            logger.info(f"[2/6] ✅ 模型目录存在，包含 {len(files)} 个文件")
            for f in files[:10]:  # 只显示前10个
                logger.info(f"       - {f}")
            
            # ═══════════════════════════════════════════════════════════════
            # Step 3: 加载模型 (这是最耗时的步骤)
            # ═══════════════════════════════════════════════════════════════
            logger.info("[3/6] 加载 CosyVoice2 模型 (fp16=True)...")
            logger.info("       ⏳ 这一步可能需要 1-3 分钟...")
            step_start = time.time()
            
            # 使用线程来打印心跳，确保能追踪进度
            import threading
            loading_done = threading.Event()
            
            def heartbeat():
                """每10秒打印一次心跳，证明没有卡死"""
                elapsed = 0
                while not loading_done.is_set():
                    time.sleep(10)
                    elapsed += 10
                    if not loading_done.is_set():
                        logger.info(f"       💓 模型加载中... ({elapsed}s)")
            
            heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
            heartbeat_thread.start()
            
            try:
                # 加载模型
                self.model = CosyVoice2(MODEL_DIR, fp16=True)
            finally:
                loading_done.set()
            
            logger.info(f"[3/6] ✅ 模型加载完成 ({time.time()-step_start:.1f}s)")
            
            # ═══════════════════════════════════════════════════════════════
            # Step 4: 获取说话人列表
            # ═══════════════════════════════════════════════════════════════
            logger.info("[4/6] 获取可用说话人...")
            speakers = self.model.list_available_spks()
            logger.info(f"[4/6] ✅ 可用说话人: {speakers}")
            
            if speakers:
                self.default_speaker = speakers[0]
                logger.info(f"[4/6] 默认说话人: {self.default_speaker}")
            else:
                logger.warning("[4/6] ⚠️ 没有可用说话人！")
            
            # ═══════════════════════════════════════════════════════════════
            # Step 5: 创建重采样器
            # ═══════════════════════════════════════════════════════════════
            logger.info(f"[5/6] 创建重采样器 ({NATIVE_SAMPLE_RATE} -> {TARGET_SAMPLE_RATE})...")
            self.resampler = torchaudio.transforms.Resample(
                NATIVE_SAMPLE_RATE, TARGET_SAMPLE_RATE
            ).cuda()
            logger.info("[5/6] ✅ 重采样器创建完成")
            
            # ═══════════════════════════════════════════════════════════════
            # Step 6: 预热推理
            # ═══════════════════════════════════════════════════════════════
            logger.info("[6/6] 预热推理中...")
            step_start = time.time()
            
            if self.default_speaker:
                for _ in self.model.inference_sft("预热测试", self.default_speaker, stream=False):
                    pass
                warmup_time = (time.time() - step_start) * 1000
                logger.info(f"[6/6] ✅ 预热完成 ({warmup_time:.0f}ms)")
            else:
                logger.warning("[6/6] ⚠️ 跳过预热（无默认说话人）")
            
            # ═══════════════════════════════════════════════════════════════
            # 完成
            # ═══════════════════════════════════════════════════════════════
            self.ready = True
            logger.info("=" * 70)
            logger.info("✅✅✅ CosyVoice2 初始化完成！服务已就绪！✅✅✅")
            logger.info("=" * 70)
            
        except Exception as e:
            logger.error("=" * 70)
            logger.error(f"❌❌❌ CosyVoice2 初始化失败: {e}")
            logger.error("=" * 70)
            import traceback
            logger.error(traceback.format_exc())
            
    def synthesize(self, text: str, speaker_id: str = None, speed: float = 1.0) -> bytes:
        """非流式合成"""
        if not self.ready:
            return b''
            
        speaker = speaker_id or self.default_speaker
        start_time = time.time()
        
        audio_chunks = []
        for output in self.model.inference_sft(text, speaker, stream=False, speed=speed):
            audio = output['tts_speech']
            audio_chunks.append(audio)
            
        if not audio_chunks:
            return b''
            
        # 合并音频
        full_audio = torch.cat(audio_chunks, dim=1)
        
        # 重采样到 44.1kHz
        if full_audio.device.type != 'cuda':
            full_audio = full_audio.cuda()
        resampled = self.resampler(full_audio)
        
        # 转换为 16-bit PCM
        audio_np = (resampled.cpu().numpy() * 32767).astype(np.int16)
        
        # 计算 RTF
        audio_duration = len(audio_np.flatten()) / TARGET_SAMPLE_RATE
        total_time = time.time() - start_time
        rtf = total_time / audio_duration
        
        if rtf > 0.1:
            logger.warning(f"⚠️ RTF={rtf:.3f} > 0.1，性能警告！")
        else:
            logger.info(f"合成完成: {len(text)}字, {total_time*1000:.0f}ms, RTF={rtf:.3f}")
        
        # 创建 WAV
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(TARGET_SAMPLE_RATE)
            wf.writeframes(audio_np.tobytes())
        return buffer.getvalue()
        
    def synthesize_stream(self, text: str, speaker_id: str = None, speed: float = 1.0):
        """真流式合成 - 每生成一段就立即返回"""
        if not self.ready:
            return
            
        speaker = speaker_id or self.default_speaker
        start_time = time.time()
        chunk_count = 0
        total_audio_duration = 0
        
        logger.info(f"开始流式合成: {text[:20]}...")
        
        # 使用 stream=True 进行真流式
        for output in self.model.inference_sft(text, speaker, stream=True, speed=speed):
            chunk_start = time.time()
            audio = output['tts_speech']
            
            # 重采样到 44.1kHz
            if audio.device.type != 'cuda':
                audio = audio.cuda()
            resampled = self.resampler(audio)
            
            # 转换为 16-bit PCM
            audio_np = (resampled.cpu().numpy() * 32767).astype(np.int16).flatten()
            
            chunk_count += 1
            chunk_duration = len(audio_np) / TARGET_SAMPLE_RATE
            total_audio_duration += chunk_duration
            
            # 计算当前 chunk 的 RTF
            chunk_time = time.time() - chunk_start
            chunk_rtf = chunk_time / chunk_duration if chunk_duration > 0 else 0
            
            if chunk_count == 1:
                ttfa = (time.time() - start_time) * 1000
                logger.info(f"TTFA: {ttfa:.0f}ms")
                
            if chunk_rtf > 0.1:
                logger.warning(f"⚠️ Chunk {chunk_count}: RTF={chunk_rtf:.3f} > 0.1")
                
            yield audio_np.tobytes()
            
        # 总结
        total_time = time.time() - start_time
        overall_rtf = total_time / total_audio_duration if total_audio_duration > 0 else 0
        logger.info(f"流式合成完成: {chunk_count} chunks, 总RTF={overall_rtf:.3f}")

    def synthesize_zero_shot(self, text: str, prompt_wav_path: str, prompt_text: str, speed: float = 1.0) -> bytes:
        """Zero-shot 克隆合成（非流式）"""
        if not self.ready:
            logger.error("服务未就绪")
            return b''
        
        logger.info(f"Zero-shot 合成: '{text[:30]}...' (参考: '{prompt_text[:20]}...')")
        start_time = time.time()
        
        try:
            audio_chunks = []
            for output in self.model.inference_zero_shot(
                text, 
                prompt_text, 
                prompt_wav_path, 
                stream=False, 
                speed=speed
            ):
                audio = output['tts_speech']
                audio_chunks.append(audio)
            
            if not audio_chunks:
                logger.error("没有生成任何音频")
                return b''
            
            # 合并音频
            full_audio = torch.cat(audio_chunks, dim=1)
            
            # 重采样到 44.1kHz
            if full_audio.device.type != 'cuda':
                full_audio = full_audio.cuda()
            resampled = self.resampler(full_audio)
            
            # 转换为 16-bit PCM
            audio_np = (resampled.cpu().numpy() * 32767).astype(np.int16)
            
            # 计算 RTF
            audio_duration = len(audio_np.flatten()) / TARGET_SAMPLE_RATE
            total_time = time.time() - start_time
            rtf = total_time / audio_duration if audio_duration > 0 else 0
            
            logger.info(f"✅ Zero-shot 完成: {len(text)}字, {total_time*1000:.0f}ms, 音频{audio_duration:.2f}s, RTF={rtf:.3f}")
            
            if rtf > 0.1:
                logger.warning(f"⚠️ RTF={rtf:.3f} > 0.1，性能警告！")
            
            # 创建 WAV
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(TARGET_SAMPLE_RATE)
                wf.writeframes(audio_np.tobytes())
            return buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Zero-shot 合成失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return b''

    def synthesize_zero_shot_stream(self, text: str, prompt_wav_path: str, prompt_text: str, speed: float = 1.0):
        """Zero-shot 克隆合成（流式）"""
        if not self.ready:
            logger.error("服务未就绪")
            return
        
        logger.info(f"Zero-shot 流式合成: '{text[:30]}...'")
        start_time = time.time()
        chunk_count = 0
        total_audio_duration = 0
        
        try:
            for output in self.model.inference_zero_shot(
                text, 
                prompt_text, 
                prompt_wav_path, 
                stream=True, 
                speed=speed
            ):
                chunk_start = time.time()
                audio = output['tts_speech']
                
                # 重采样到 44.1kHz
                if audio.device.type != 'cuda':
                    audio = audio.cuda()
                resampled = self.resampler(audio)
                
                # 转换为 16-bit PCM
                audio_np = (resampled.cpu().numpy() * 32767).astype(np.int16).flatten()
                
                chunk_count += 1
                chunk_duration = len(audio_np) / TARGET_SAMPLE_RATE
                total_audio_duration += chunk_duration
                
                if chunk_count == 1:
                    ttfa = (time.time() - start_time) * 1000
                    logger.info(f"⚡ TTFA: {ttfa:.0f}ms")
                
                yield audio_np.tobytes()
                
            # 总结
            total_time = time.time() - start_time
            overall_rtf = total_time / total_audio_duration if total_audio_duration > 0 else 0
            logger.info(f"✅ Zero-shot 流式完成: {chunk_count} chunks, 总RTF={overall_rtf:.3f}")
            
        except Exception as e:
            logger.error(f"Zero-shot 流式合成失败: {e}")
            import traceback
            logger.error(traceback.format_exc())

# ═══════════════════════════════════════════════════════════════════════════════
# 全局实例
# ═══════════════════════════════════════════════════════════════════════════════
service = CosyVoice2Service()

# ═══════════════════════════════════════════════════════════════════════════════
# FastAPI 应用
# ═══════════════════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"🎤 CosyVoice2 服务启动中 (端口 {SERVICE_PORT})...")
    service.initialize()
    yield
    logger.info("🛑 CosyVoice2 服务关闭")

app = FastAPI(title="CosyVoice2 TTS Service", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {
        "service": "mouth-cosyvoice2",
        "status": "ready" if service.ready else "loading",
        "model": "CosyVoice2 0.5B (fp16=True)",
        "sample_rate": TARGET_SAMPLE_RATE,
        "native_sample_rate": NATIVE_SAMPLE_RATE,
        "speakers": service.model.list_available_spks() if service.model else [],
        "default_speaker": service.default_speaker
    }

@app.post("/tts")
async def tts(
    text: str = Form(...),
    prompt_text: str = Form(...),
    prompt_audio: UploadFile = File(...),
    speed: float = Form(1.0)
):
    """
    Zero-shot TTS（非流式）
    
    需要上传参考音频和参考文本来克隆声音
    """
    if not service.ready:
        logger.error("服务未就绪")
        return Response(content=b'Service not ready', status_code=503)
    
    try:
        # 保存上传的音频到临时文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            content = await prompt_audio.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        logger.info(f"接收到 TTS 请求: text='{text[:30]}...', prompt_text='{prompt_text[:20]}...', prompt_audio={len(content)} bytes")
        
        # 合成
        audio = service.synthesize_zero_shot(text, tmp_path, prompt_text, speed)
        
        # 清理临时文件
        os.unlink(tmp_path)
        
        if not audio:
            return Response(content=b'Synthesis failed', status_code=500)
        
        return Response(content=audio, media_type="audio/wav")
        
    except Exception as e:
        logger.error(f"TTS 请求处理失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return Response(content=str(e).encode(), status_code=500)

@app.post("/tts/stream")
async def tts_stream(
    text: str = Form(...),
    prompt_text: str = Form(...),
    prompt_audio: UploadFile = File(...),
    speed: float = Form(1.0)
):
    """
    Zero-shot TTS（真流式）- 每生成约 50ms 就立即返回
    """
    if not service.ready:
        logger.error("服务未就绪")
        return Response(content=b'Service not ready', status_code=503)
    
    try:
        # 保存上传的音频到临时文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            content = await prompt_audio.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        logger.info(f"接收到流式 TTS 请求: text='{text[:30]}...', prompt_audio={len(content)} bytes")
        
        def generate():
            try:
                for chunk in service.synthesize_zero_shot_stream(text, tmp_path, prompt_text, speed):
                    yield chunk
            finally:
                # 清理临时文件
                try:
                    os.unlink(tmp_path)
                except:
                    pass
        
        return StreamingResponse(
            generate(),
            media_type="audio/pcm",
            headers={
                "X-Sample-Rate": str(TARGET_SAMPLE_RATE),
                "X-Channels": "1",
                "X-Bit-Depth": "16"
            }
        )
        
    except Exception as e:
        logger.error(f"流式 TTS 请求处理失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return Response(content=str(e).encode(), status_code=500)

@app.get("/speakers")
async def list_speakers():
    """获取可用说话人列表"""
    if not service.model:
        return {"speakers": []}
    return {"speakers": service.model.list_available_spks()}

# ═══════════════════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    
    # 清理端口
    kill_port(SERVICE_PORT)
    
    logger.info(f"🚀 启动 CosyVoice2 服务 (端口 {SERVICE_PORT})...")
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT, log_level="info")
