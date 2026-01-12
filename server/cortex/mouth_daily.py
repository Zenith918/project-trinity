"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  👄 CORTEX-MOUTH-DAILY (端口 9003)                                            ║
║  VoxCPM 1.5 - 极致低延迟配置                                                  ║
║                                                                              ║
║  🔥 optimize=True + mode="default" = TTFA ~285ms                              ║
║  💡 首次启动需要 ~10 分钟 JIT 编译，之后稳定在 ~285ms                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════╗
║  ⚠️⚠️⚠️ 警告：绝对不要修改以下配置！⚠️⚠️⚠️                                      ║
║                                                                              ║
║  1. optimize=True  - 必须为 True，否则 TTFA 会从 285ms 退化到 450ms            ║
║  2. mode="default" - VoxCPM 源码已修改，不要改回 "reduce-overhead"             ║
║                                                                              ║
║  如果你看到 CUDA Graph 相关错误，问题在 VoxCPM 源码，不是这里！                 ║
║  解决方案：修改 /usr/local/lib/python3.11/dist-packages/voxcpm/model/voxcpm.py ║
║  将 mode="reduce-overhead" 改为 mode="default"                                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import signal
import subprocess

# ═══════════════════════════════════════════════════════════════════════════════
# 🔧 启动前自动清理端口（防止 "Address already in use" 错误）
# ═══════════════════════════════════════════════════════════════════════════════
SERVICE_PORT = 9003

def kill_port(port: int):
    """杀掉占用指定端口的进程"""
    try:
        # 使用 fuser 查找并杀掉占用端口的进程
        result = subprocess.run(
            f"fuser -k {port}/tcp 2>/dev/null || true",
            shell=True, capture_output=True, text=True
        )
        # 备用方案：使用 ss + kill
        result = subprocess.run(
            f"ss -tlnp 2>/dev/null | grep ':{port}' | awk '{{print $NF}}' | grep -oP 'pid=\\K[0-9]+' | xargs -r kill -9 2>/dev/null || true",
            shell=True, capture_output=True, text=True
        )
    except Exception:
        pass

# 启动前先清理端口
kill_port(SERVICE_PORT)

# ═══════════════════════════════════════════════════════════════════════════════
# 🔑 关键：在导入 torch 之前禁用 CUDA Graph（虽然已改 VoxCPM 源码，但双重保险）
# ═══════════════════════════════════════════════════════════════════════════════
os.environ['TORCHINDUCTOR_CUDAGRAPHS'] = '0'

import torch
try:
    torch._inductor.config.triton.cudagraphs = False
except AttributeError:
    pass  # torch 2.4.x 不需要这个设置，环境变量已生效

import io
import wave
import numpy as np
from loguru import logger
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response, FileResponse
from contextlib import asynccontextmanager
import time

mouth = None


class DailyMouthHandler:
    """
    VoxCPM 1.5 处理器
    
    ⚠️ 重要配置（不要修改）：
    - optimize=True  → 启用 torch.compile，TTFA ~285ms
    - optimize=False → 禁用优化，TTFA 退化到 ~450ms（慢 37%）
    """
    
    def __init__(self):
        self.model = None
        self.is_ready = False
        # 🔥 VoxCPM 1.5 使用 44.1kHz 高保真采样率！不是 24kHz！
        self.sample_rate = 44100
        # steps=2 实现 RTF < 1 (实时流畅播放)，steps=4 音质更好但会卡顿
        self.config = {"steps": 2, "cfg_value": 2.0}
        
        # 音色 Prompt 配置
        # VoxCPM 要求 prompt_wav_path 和 prompt_text 必须同时提供或同时为空
        # 使用 44.1kHz 重采样版本，匹配 VoxCPM 1.5 输出采样率
        self.default_prompt_wav = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "assets", "prompt_female_44k.wav"
        )
        # prompt 音频内容（通过 ASR 识别）
        self.default_prompt_text = "希望你以后能够做的比我还好哟"
        
    async def initialize(self):
        logger.info("=" * 60)
        logger.info("正在初始化 VoxCPM 1.5 (optimize=True)...")
        logger.info("⚠️ 首次启动需要 ~10 分钟 JIT 编译，请耐心等待！")
        logger.info("=" * 60)
        
        try:
            from voxcpm import VoxCPM
            
            # ═══════════════════════════════════════════════════════════════
            # 🚨🚨🚨 绝对不要把 optimize 改成 False！🚨🚨🚨
            # 
            # optimize=True  → TTFA ~285ms ✅
            # optimize=False → TTFA ~450ms ❌ (慢 37%)
            #
            # 如果遇到 CUDA Graph 错误，修改 VoxCPM 源码，不要改这里！
            # ═══════════════════════════════════════════════════════════════
            OPTIMIZE_ENABLED = True  # 🚨 不要改成 False！
            assert OPTIMIZE_ENABLED is True, "❌ optimize 必须为 True！不要改成 False！"
            
            self.model = VoxCPM.from_pretrained(
                hf_model_id="openbmb/VoxCPM1.5",
                load_denoiser=False,
                optimize=OPTIMIZE_ENABLED,  # 🚨 必须为 True
            )
            
            # 预热 1: 非流式
            logger.info("预热 1/2: 非流式推理...")
            _ = self.model.generate(
                text="预热",
                inference_timesteps=self.config["steps"],
                cfg_value=self.config["cfg_value"],
            )
            
            # 预热 2: 流式 (触发完整 JIT)
            logger.info("预热 2/2: 流式推理 (触发 JIT 编译)...")
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

    def synthesize(self, text, inference_timesteps=None, cfg_value=None, 
                   prompt_wav_path=None, prompt_text=None):
        """
        非流式语音合成
        
        Args:
            text: 要合成的文本
            inference_timesteps: 扩散步数 (2-10, 越高越清晰但越慢)
            cfg_value: CFG 值 (1.0-3.0, 越高越清晰)
            prompt_wav_path: 参考音频路径 (用于克隆音色)
            prompt_text: 参考音频对应的文本 (可选)
        """
        if not self.is_ready:
            return b""
        steps = inference_timesteps or self.config["steps"]
        cfg = cfg_value or self.config["cfg_value"]
        prompt_wav = prompt_wav_path or self.default_prompt_wav
        prompt_txt = prompt_text or self.default_prompt_text
        
        try:
            start = time.time()
            actual_prompt_wav = prompt_wav if os.path.exists(prompt_wav) else None
            actual_prompt_txt = prompt_txt if prompt_txt else None
            
            # 使用 prompt 音频克隆音色
            # 注意：core.py 的参数是 "text" 不是 "target_text"
            audio = self.model.generate(
                text=text, 
                cfg_value=cfg, 
                inference_timesteps=steps,
                prompt_wav_path=actual_prompt_wav,
                prompt_text=actual_prompt_txt,
            )
            logger.info(f"生成: {len(text)}字, {(time.time()-start)*1000:.0f}ms, prompt={os.path.basename(prompt_wav) if prompt_wav else 'none'}")
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
            import traceback
            logger.error(traceback.format_exc())
            return b""

    def synthesize_stream(self, text, inference_timesteps=None, cfg_value=None,
                          prompt_wav_path=None, prompt_text=None):
        """
        流式语音合成 (带 Early Stopping + Chunk 合并)
        
        优化策略：
        1. Early Stopping: 根据文本长度预估最大 chunk 数，防止 AR 幻觉循环
        2. Chunk 合并: 每 2 个 chunk 合并后再 yield，减少网络 IO
        """
        if not self.is_ready:
            yield b""
            return
        steps = inference_timesteps or self.config["steps"]
        cfg = cfg_value or self.config["cfg_value"]
        prompt_wav = prompt_wav_path or self.default_prompt_wav
        prompt_txt = prompt_text or self.default_prompt_text
        
        # 🔧 Early Stopping: 1个汉字约 3-5 个 token (480-800ms)
        # 每个 chunk = 160ms = 1 token，给宽松上限：字数 * 8
        text_len = len(text.replace(" ", ""))
        max_chunks = max(15, text_len * 8)  # 最少 15 个 chunk (2.4s)
        
        # 🔧 Chunk 合并: 减少 IO 次数
        MERGE_COUNT = 2  # 每 2 个 chunk 合并发送
        
        try:
            start = time.time()
            first = True
            chunk_count = 0
            pending_chunks = []  # 待合并的 chunk 缓冲
            
            for chunk in self.model.generate_streaming(
                text=text, 
                cfg_value=cfg, 
                inference_timesteps=steps,
                prompt_wav_path=prompt_wav if os.path.exists(prompt_wav) else None,
                prompt_text=prompt_txt if prompt_txt else None,
            ):
                chunk_count += 1
                
                if first:
                    logger.info(f"TTFA: {(time.time()-start)*1000:.0f}ms, max_chunks={max_chunks}")
                    first = False
                
                # Early Stopping: 防止 AR 幻觉循环
                if chunk_count > max_chunks:
                    logger.warning(f"⚠️ Early Stop: 已达 {chunk_count} chunks (上限 {max_chunks})，强制截断")
                    # 输出剩余缓冲
                    if pending_chunks:
                        merged = np.concatenate(pending_chunks)
                        yield (merged * 32767).astype(np.int16).tobytes()
                    break
                
                pending_chunks.append(chunk)
                
                # Chunk 合并: 积攒够了再发送
                if len(pending_chunks) >= MERGE_COUNT:
                    merged = np.concatenate(pending_chunks)
                    yield (merged * 32767).astype(np.int16).tobytes()
                    pending_chunks = []
            
            # 输出最后的缓冲
            if pending_chunks:
                merged = np.concatenate(pending_chunks)
                yield (merged * 32767).astype(np.int16).tobytes()
                
            logger.info(f"流式完成: {chunk_count} chunks, {(time.time()-start)*1000:.0f}ms")
                    
        except Exception as e:
            logger.error(f"流式失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            yield b""


@asynccontextmanager
async def lifespan(app: FastAPI):
    global mouth
    logger.info(f"👄 Cortex-Mouth-Daily 启动中 (端口 {SERVICE_PORT})...")
    mouth = DailyMouthHandler()
    await mouth.initialize()
    if mouth.is_ready:
        logger.success(f"✅ Mouth-Daily 就绪 (端口 {SERVICE_PORT})")
    yield
    logger.info("🛑 Mouth-Daily 关闭")


app = FastAPI(lifespan=lifespan, title="Cortex-Mouth-Daily")

# 托管静态文件 (用于测试页面)
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/test", StaticFiles(directory=static_dir, html=True), name="static")

@app.get("/")
async def root():
    return FileResponse(os.path.join(static_dir, "index.html"))

# CORS 支持 - 允许前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    prompt_exists = mouth and os.path.exists(mouth.default_prompt_wav) if mouth else False
    return {
        "service": "mouth-daily",
        "status": "ok" if mouth and mouth.is_ready else "loading",
        "model": "VoxCPM 1.5 (optimize=True, mode=default)",
        "sample_rate": 44100,  # VoxCPM 1.5 高保真采样率
        "ttfa_target": "~285ms",
        "config": mouth.config if mouth else {},
        "voice_prompt": {
            "enabled": prompt_exists,
            "path": mouth.default_prompt_wav if mouth else None,
        }
    }


@app.post("/tts")
async def tts(request: dict):
    """
    非流式 TTS
    
    Request body:
        text: 要合成的文本 (必填)
        inference_timesteps: 步数 (可选, 默认 4, 范围 2-10)
        cfg_value: CFG 值 (可选, 默认 2.0, 范围 1.0-3.0)
        prompt_wav_path: 参考音频路径 (可选, 默认使用内置女声)
        prompt_text: 参考音频文本 (可选)
    """
    if not mouth or not mouth.is_ready:
        return {"error": "Not ready"}
    text = request.get("text", "")
    if not text:
        return {"error": "text required"}
    audio = mouth.synthesize(
        text, 
        request.get("inference_timesteps"), 
        request.get("cfg_value"),
        request.get("prompt_wav_path"),
        request.get("prompt_text"),
    )
    if not audio:
        return {"error": "failed"}
    return Response(content=audio, media_type="audio/wav")


@app.post("/tts/stream")
async def tts_stream(request: dict):
    """
    流式 TTS
    
    Request body:
        text: 要合成的文本 (必填)
        inference_timesteps: 步数 (可选, 默认 4, 范围 2-10)
        cfg_value: CFG 值 (可选, 默认 2.0, 范围 1.0-3.0)
        prompt_wav_path: 参考音频路径 (可选, 默认使用内置女声)
        prompt_text: 参考音频文本 (可选)
    """
    if not mouth or not mouth.is_ready:
        return {"error": "Not ready"}
    text = request.get("text", "")
    if not text:
        return {"error": "text required"}
    return StreamingResponse(
        mouth.synthesize_stream(
            text, 
            request.get("inference_timesteps"), 
            request.get("cfg_value"),
            request.get("prompt_wav_path"),
            request.get("prompt_text"),
        ),
        media_type="audio/pcm",
        headers={"X-Sample-Rate": "44100"}  # VoxCPM 1.5 高保真采样率
    )


if __name__ == "__main__":
    import uvicorn
    # 再次确保端口清理
    kill_port(SERVICE_PORT)
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT)
