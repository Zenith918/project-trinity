"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  💋 CORTEX-MOUTH-EMOTION (端口 9004)                                          ║
║  IndexTTS 2.5 - 情感增强 TTS 服务                                              ║
║                                                                              ║
║  🎭 特性：                                                                    ║
║    - 8维情感向量控制 (happy/angry/sad/afraid/disgusted/melancholic/surprised/calm)║
║    - 自动文本情感分析 (use_emo_text=True)                                      ║
║    - torch.compile 加速                                                       ║
║    - 22kHz -> 44.1kHz 实时重采样                                               ║
║    - AR 幻觉截断保护                                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys

# 🔑 设置 IndexTTS 模块路径
INDEXTTS_PATH = "/workspace/models/IndexTTS2.5/index-tt2.5"
sys.path.insert(0, INDEXTTS_PATH)
os.chdir(INDEXTTS_PATH)  # IndexTTS 依赖相对路径加载 checkpoints

# 禁用 CUDA Graph (与流式不兼容)
os.environ['TORCHINDUCTOR_CUDAGRAPHS'] = '0'

import torch
# torch 2.4.x API 兼容性处理
try:
    torch._inductor.config.triton.cudagraphs = False
except AttributeError:
    pass  # torch 2.4.x 不需要这个设置

import io
import wave
import time
import numpy as np
import torchaudio
from loguru import logger
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from typing import Optional, List

# ============================================================================
# 配置常量
# ============================================================================
SERVICE_PORT = 9004
MODEL_DIR = os.path.join(INDEXTTS_PATH, "checkpoints")
CFG_PATH = os.path.join(MODEL_DIR, "config.yaml")

# 采样率配置
NATIVE_SAMPLE_RATE = 22050  # IndexTTS 原生采样率
OUTPUT_SAMPLE_RATE = 44100  # 统一输出采样率

# AR 幻觉截断阈值 (tokens per character)
MAX_TOKENS_PER_CHAR = 10

mouth = None


class EmotionMouthHandler:
    """IndexTTS 2.5 处理器 - 情感增强版"""
    
    def __init__(self):
        self.model = None
        self.is_ready = False
        self.native_sample_rate = NATIVE_SAMPLE_RATE
        self.output_sample_rate = OUTPUT_SAMPLE_RATE
        self.resampler = None  # 延迟初始化
        
        # 默认 prompt 音频 (使用现有的)
        self.default_prompt_wav = "/workspace/project-trinity/project-trinity/assets/prompt_female.wav"
        
        # 情感向量顺序 (与 IndexTTS2 内部一致)
        self.emotion_keys = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]
        
    async def initialize(self):
        logger.info("=" * 60)
        logger.info("正在初始化 IndexTTS 2.5 (情感增强版)...")
        logger.info(f"模型目录: {MODEL_DIR}")
        logger.info(f"配置文件: {CFG_PATH}")
        logger.info("⚠️ 首次启动需要加载多个模型 + torch.compile 预热，请耐心等待！")
        logger.info("=" * 60)
        
        try:
            from indextts.infer_v2 import IndexTTS2
            
            start_time = time.time()
            
            # 🔥 修复版配置：避免 CUDA Graph 冲突
            # ============================================================
            # 2026-01-11 修复：静默崩溃问题
            # 根因：use_accel (GPT CUDA Graph) 与 use_torch_compile 冲突
            # 方案：禁用 torch.compile，保留模型原生 accel 引擎
            # ============================================================
            
            # 步骤 4: 显存防碎片 (在加载前清理)
            torch.cuda.empty_cache()
            
            self.model = IndexTTS2(
                cfg_path=CFG_PATH,
                model_dir=MODEL_DIR,
                use_fp16=True,           # 🔑 FP16 加速
                use_cuda_kernel=True,   # BigVGAN CUDA kernel (需要 Ninja 编译，禁用)
                use_deepspeed=False,     # 单用户不需要
                use_accel=True,          # 🔑 保留：模型原生 CUDA Graph (GPT 加速)
                use_torch_compile=False, # 🔑 禁用：防止与 accel 冲突
                device="cuda:0"
            )
            
            # 加载后再次清理碎片
            torch.cuda.empty_cache()
            
            load_time = time.time() - start_time
            logger.info(f"模型加载完成，耗时 {load_time:.1f}s")
            
            # 初始化重采样器 (22kHz -> 44.1kHz)
            self.resampler = torchaudio.transforms.Resample(
                orig_freq=self.native_sample_rate,
                new_freq=self.output_sample_rate
            ).cuda()
            logger.info(f"重采样器就绪: {self.native_sample_rate}Hz -> {self.output_sample_rate}Hz")
            
            # 🔥 预热推理 (触发 torch.compile JIT)
            logger.info("预热推理中 (触发 JIT 编译，可能需要 30-60 秒)...")
            warmup_start = time.time()
            
            # 使用一个简短的"。"来预热
            _ = list(self.model.infer_generator(
                spk_audio_prompt=self.default_prompt_wav,
                text="。",  # 最短的合法输入
                output_path=None,
                stream_return=False,
                verbose=False
            ))
            
            warmup_time = time.time() - warmup_start
            logger.success(f"✅ 预热完成，耗时 {warmup_time:.1f}s")
            
            self.is_ready = True
            total_time = time.time() - start_time
            logger.success(f"✅ IndexTTS 2.5 初始化完成！总耗时 {total_time:.1f}s")
            return True
            
        except Exception as e:
            logger.error(f"IndexTTS 初始化失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _resample_to_44k(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """将音频从 22kHz 重采样到 44.1kHz"""
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        return self.resampler(audio_tensor.cuda()).cpu()

    def synthesize(self, text: str, 
                   prompt_wav_path: Optional[str] = None,
                   emo_vector: Optional[List[float]] = None,
                   use_emo_text: bool = False) -> bytes:
        """非流式合成"""
        if not self.is_ready:
            return b""
        
        prompt_wav = prompt_wav_path or self.default_prompt_wav
        
        # AR 幻觉保护：预估最大合理长度
        text_len = len(text.replace(" ", ""))
        max_mel_tokens = min(1500, max(200, text_len * MAX_TOKENS_PER_CHAR))
        
        try:
            start = time.time()
            
            # 调用 IndexTTS2 推理
            result = list(self.model.infer_generator(
                spk_audio_prompt=prompt_wav,
                text=text,
                output_path=None,
                emo_vector=emo_vector,
                use_emo_text=use_emo_text,
                stream_return=False,
                verbose=False,
                max_mel_tokens=max_mel_tokens,  # 幻觉截断
            ))
            
            if not result:
                logger.warning("IndexTTS 返回空结果")
                return b""
            
            # result[-1] 是 (sample_rate, wav_data) 元组
            sr, wav_data = result[-1]
            
            # 重采样到 44.1kHz
            wav_tensor = torch.from_numpy(wav_data.T.astype(np.float32) / 32767.0)
            wav_44k = self._resample_to_44k(wav_tensor)
            
            # 转换为 WAV 字节
            audio_int16 = (wav_44k.squeeze().numpy() * 32767).astype(np.int16)
            buf = io.BytesIO()
            with wave.open(buf, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.output_sample_rate)
                wf.writeframes(audio_int16.tobytes())
            
            elapsed = (time.time() - start) * 1000
            logger.info(f"合成完成: {text_len}字, {elapsed:.0f}ms, emo={emo_vector or 'auto' if use_emo_text else 'none'}")
            
            return buf.getvalue()
            
        except Exception as e:
            logger.error(f"合成失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return b""

    def synthesize_stream(self, text: str,
                          prompt_wav_path: Optional[str] = None,
                          emo_vector: Optional[List[float]] = None,
                          use_emo_text: bool = False):
        """流式合成 (Generator)"""
        if not self.is_ready:
            yield b""
            return
        
        prompt_wav = prompt_wav_path or self.default_prompt_wav
        
        # AR 幻觉保护
        text_len = len(text.replace(" ", ""))
        max_mel_tokens = min(1500, max(200, text_len * MAX_TOKENS_PER_CHAR))
        max_chunks = max(10, text_len * 5)  # 每个字最多 5 个 chunk
        
        try:
            start = time.time()
            first_chunk = True
            chunk_count = 0
            
            for chunk in self.model.infer_generator(
                spk_audio_prompt=prompt_wav,
                text=text,
                output_path=None,
                emo_vector=emo_vector,
                use_emo_text=use_emo_text,
                stream_return=True,
                verbose=False,
                max_mel_tokens=max_mel_tokens,
            ):
                chunk_count += 1
                
                # 幻觉截断
                if chunk_count > max_chunks:
                    logger.warning(f"⚠️ Early Stop: 检测到流式幻觉，强制截断！已输出 {chunk_count} chunks")
                    break
                
                if first_chunk:
                    ttfa = (time.time() - start) * 1000
                    logger.info(f"TTFA: {ttfa:.0f}ms")
                    first_chunk = False
                
                # chunk 是 torch.Tensor, shape [1, samples] 或 [samples]
                if isinstance(chunk, torch.Tensor):
                    # 重采样到 44.1kHz
                    wav_44k = self._resample_to_44k(chunk.float() / 32767.0)
                    audio_int16 = (wav_44k.squeeze().numpy() * 32767).astype(np.int16)
                    yield audio_int16.tobytes()
                    
        except Exception as e:
            logger.error(f"流式合成失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            yield b""


@asynccontextmanager
async def lifespan(app: FastAPI):
    global mouth
    logger.info(f"💋 Cortex-Mouth-Emotion 启动中 (端口 {SERVICE_PORT})...")
    mouth = EmotionMouthHandler()
    success = await mouth.initialize()
    if success:
        logger.success(f"✅ Mouth-Emotion 就绪 (端口 {SERVICE_PORT})")
    else:
        logger.error("❌ Mouth-Emotion 初始化失败")
    yield
    logger.info("🛑 Mouth-Emotion 关闭")


app = FastAPI(lifespan=lifespan, title="Cortex-Mouth-Emotion (IndexTTS 2.5)")

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
        "service": "mouth-emotion",
        "status": "ok" if mouth and mouth.is_ready else "loading",
        "model": "IndexTTS 2.5 (use_fp16=True, torch_compile=True)",
        "sample_rate": OUTPUT_SAMPLE_RATE,
        "native_sample_rate": NATIVE_SAMPLE_RATE,
        "emotion_keys": mouth.emotion_keys if mouth else [],
        "voice_prompt": mouth.default_prompt_wav if mouth else None,
    }


@app.post("/tts")
async def tts(request: dict):
    """
    非流式 TTS 接口
    
    请求体:
    {
        "text": "要合成的文本",
        "prompt_wav_path": "可选，参考音频路径",
        "emo_vector": [0.5, 0, 0, 0, 0, 0, 0, 0.5],  // 可选，8维情感向量
        "use_emo_text": false  // 可选，是否从文本自动推断情感
    }
    """
    if not mouth or not mouth.is_ready:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    text = request.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="text required")
    
    audio = mouth.synthesize(
        text=text,
        prompt_wav_path=request.get("prompt_wav_path"),
        emo_vector=request.get("emo_vector"),
        use_emo_text=request.get("use_emo_text", False),
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
            prompt_wav_path=request.get("prompt_wav_path"),
            emo_vector=request.get("emo_vector"),
            use_emo_text=request.get("use_emo_text", False),
        ),
        media_type="audio/pcm",
        headers={"X-Sample-Rate": str(OUTPUT_SAMPLE_RATE)}
    )


@app.post("/analyze_emotion")
async def analyze_emotion(request: dict):
    """
    分析文本情感
    
    请求体: {"text": "要分析的文本"}
    返回: {"happy": 0.5, "angry": 0, ...}
    """
    if not mouth or not mouth.is_ready:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    text = request.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="text required")
    
    try:
        emo_dict = mouth.model.qwen_emo.inference(text)
        return emo_dict
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    from server.utils.port_utils import kill_port
    
    kill_port(SERVICE_PORT)
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT)

