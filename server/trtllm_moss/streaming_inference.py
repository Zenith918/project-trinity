"""
MOSS-Speech 流式推理
=====================

研究员方案:
- first_chunk_size=5 实现极速首包响应
- Token 流式喂给声码器
- 异步并行: 解码 Chunk N 时生成 Chunk N+1

目标:
- TTFA < 300ms
- RTF < 1
"""

import os
import sys
import torch
import numpy as np
import asyncio
import time
from typing import Optional, List, Generator, AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StreamingConfig:
    """流式推理配置"""
    # 首包 chunk 大小 (研究员方案: 5 tokens)
    first_chunk_size: int = 5
    # 后续 chunk 大小
    normal_chunk_size: int = 20
    # 最大生成长度
    max_length: int = 2048
    # 采样参数
    temperature: float = 0.7
    top_p: float = 0.9
    # 音频参数
    sample_rate: int = 22050


class StreamingBuffer:
    """
    流式 Token 缓冲区
    
    研究员方案: 每 10-20ms 一片
    """
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        self._tokens: List[int] = []
        self._is_first_chunk = True
        self._total_generated = 0
    
    @property
    def chunk_size(self) -> int:
        """当前 chunk 大小 (首包更小)"""
        if self._is_first_chunk:
            return self.config.first_chunk_size
        return self.config.normal_chunk_size
    
    def add(self, token: int) -> Optional[List[int]]:
        """
        添加 token，返回完整 chunk (如果有)
        
        Returns:
            chunk tokens 或 None
        """
        self._tokens.append(token)
        self._total_generated += 1
        
        if len(self._tokens) >= self.chunk_size:
            chunk = self._tokens[:self.chunk_size]
            self._tokens = self._tokens[self.chunk_size:]
            
            if self._is_first_chunk:
                self._is_first_chunk = False
            
            return chunk
        
        return None
    
    def flush(self) -> Optional[List[int]]:
        """刷新剩余 tokens"""
        if self._tokens:
            chunk = self._tokens
            self._tokens = []
            return chunk
        return None
    
    def reset(self):
        """重置状态"""
        self._tokens = []
        self._is_first_chunk = True
        self._total_generated = 0


class MOSSSpeechStreamingInference:
    """
    MOSS-Speech 流式推理引擎
    
    架构:
    1. TRT-LLM Engine 生成 Audio Tokens (流式)
    2. StreamingBuffer 收集 chunks
    3. BigVGAN Vocoder 解码到波形
    """
    
    def __init__(
        self,
        engine_path: str,
        vocoder_path: str,
        config: Optional[StreamingConfig] = None,
        device: str = "cuda",
    ):
        self.engine_path = Path(engine_path)
        self.vocoder_path = Path(vocoder_path)
        self.config = config or StreamingConfig()
        self.device = device
        
        self._engine = None
        self._vocoder = None
        self._tokenizer = None
        self._buffer = StreamingBuffer(self.config)
        
        # 性能统计
        self._stats = {
            'ttfa_ms': 0,
            'total_time_ms': 0,
            'tokens_generated': 0,
            'audio_duration_s': 0,
        }
    
    def load(self):
        """加载模型"""
        logger.info("Loading MOSS-Speech TRT-LLM Engine...")
        self._load_engine()
        
        logger.info("Loading BigVGAN Vocoder...")
        self._load_vocoder()
        
        logger.info("✅ All models loaded")
    
    def _load_engine(self):
        """加载 TRT-LLM Engine"""
        try:
            # TRT-LLM Runner
            import tensorrt_llm
            from tensorrt_llm.runtime import ModelRunner
            
            if self.engine_path.exists():
                self._engine = ModelRunner.from_dir(str(self.engine_path))
                logger.info(f"✅ Engine loaded from {self.engine_path}")
            else:
                logger.warning(f"Engine not found: {self.engine_path}")
                logger.info("Using fallback PyTorch model")
                self._load_fallback_model()
                
        except ImportError as e:
            logger.warning(f"TRT-LLM not available: {e}")
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """加载原始 PyTorch 模型作为 fallback"""
        from transformers import AutoModel, AutoTokenizer
        
        model_path = "/workspace/models/MOSS-Speech"
        self._engine = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        logger.info("✅ Fallback PyTorch model loaded")
    
    def _load_vocoder(self):
        """加载声码器"""
        from vocoder import BigVGANVocoder
        
        self._vocoder = BigVGANVocoder(
            model_path=str(self.vocoder_path),
            use_cuda_kernel=True,
        )
        self._vocoder.load()
    
    async def generate_streaming(
        self,
        text: str,
        audio_prompt: Optional[torch.Tensor] = None,
    ) -> AsyncGenerator[np.ndarray, None]:
        """
        流式生成语音
        
        研究员方案: Token 流式喂给声码器
        
        Args:
            text: 输入文本
            audio_prompt: 可选的音频 prompt (用于克隆)
            
        Yields:
            audio_chunk: PCM 音频块
        """
        self._buffer.reset()
        self._stats = {k: 0 for k in self._stats}
        
        start_time = time.perf_counter()
        first_audio_time = None
        
        # 生成 tokens (流式)
        async for token in self._generate_tokens_async(text, audio_prompt):
            # 添加到 buffer
            chunk = self._buffer.add(token)
            
            if chunk is not None:
                # 解码 chunk 到音频
                audio = self._decode_chunk(chunk)
                
                # 记录 TTFA
                if first_audio_time is None:
                    first_audio_time = time.perf_counter()
                    self._stats['ttfa_ms'] = (first_audio_time - start_time) * 1000
                    logger.info(f"🎯 TTFA: {self._stats['ttfa_ms']:.2f}ms")
                
                yield audio
        
        # 处理剩余 tokens
        remaining = self._buffer.flush()
        if remaining:
            audio = self._decode_chunk(remaining)
            yield audio
        
        # 计算最终统计
        total_time = time.perf_counter() - start_time
        self._stats['total_time_ms'] = total_time * 1000
        self._stats['tokens_generated'] = self._buffer._total_generated
        
        # 估算音频时长 (假设 86 tokens/sec)
        self._stats['audio_duration_s'] = self._buffer._total_generated / 86
        
        rtf = total_time / max(self._stats['audio_duration_s'], 0.001)
        logger.info(f"📊 总时间: {self._stats['total_time_ms']:.2f}ms, RTF: {rtf:.2f}")
    
    async def _generate_tokens_async(
        self,
        text: str,
        audio_prompt: Optional[torch.Tensor] = None,
    ) -> AsyncGenerator[int, None]:
        """异步生成 tokens"""
        # TRT-LLM Engine 或 PyTorch fallback
        if hasattr(self._engine, 'generate_streaming'):
            # TRT-LLM 流式生成
            for token in self._engine.generate_streaming(text):
                yield token
                await asyncio.sleep(0)  # 让出控制权
        else:
            # PyTorch fallback (模拟流式)
            for token in self._generate_tokens_pytorch(text, audio_prompt):
                yield token
                await asyncio.sleep(0)
    
    def _generate_tokens_pytorch(
        self,
        text: str,
        audio_prompt: Optional[torch.Tensor] = None,
    ) -> Generator[int, None, None]:
        """PyTorch 模型生成 tokens (逐个)"""
        if self._tokenizer is None:
            # Mock tokens
            for i in range(100):
                yield i % 4096
            return
        
        # 编码输入
        inputs = self._tokenizer(text, return_tensors="pt").to(self.device)
        
        # 逐 token 生成
        past_key_values = None
        generated = []
        
        with torch.no_grad():
            for _ in range(self.config.max_length):
                outputs = self._engine(
                    **inputs if past_key_values is None else {'input_ids': inputs['input_ids'][:, -1:]},
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values
                
                # 采样
                if self.config.temperature > 0:
                    probs = torch.softmax(logits / self.config.temperature, dim=-1)
                    token = torch.multinomial(probs, num_samples=1).item()
                else:
                    token = logits.argmax(dim=-1).item()
                
                # 检查 EOS (假设 audio_eos_token_id 在配置中)
                if hasattr(self._engine.config, 'audio_eos_token_id'):
                    if token == self._engine.config.audio_eos_token_id:
                        break
                
                generated.append(token)
                yield token
                
                inputs['input_ids'] = torch.cat([
                    inputs['input_ids'],
                    torch.tensor([[token]], device=self.device)
                ], dim=1)
    
    def _decode_chunk(self, tokens: List[int]) -> np.ndarray:
        """解码 token chunk 到音频"""
        if self._vocoder is None:
            # Mock 音频
            return np.random.randn(len(tokens) * 256).astype(np.float32) * 0.01
        
        return self._vocoder.decode_tokens(tokens)
    
    def get_stats(self) -> dict:
        """获取性能统计"""
        return self._stats.copy()


# === 同步版本 (用于简单测试) ===
def generate_speech(
    engine,
    vocoder,
    text: str,
    config: Optional[StreamingConfig] = None,
) -> tuple[np.ndarray, dict]:
    """
    同步生成语音
    
    Returns:
        (audio, stats)
    """
    config = config or StreamingConfig()
    inference = MOSSSpeechStreamingInference(
        engine_path=engine,
        vocoder_path=vocoder,
        config=config,
    )
    inference._engine = engine  # 直接使用传入的 engine
    inference._vocoder = vocoder
    
    # 收集所有音频块
    audio_chunks = []
    
    async def run():
        async for chunk in inference.generate_streaming(text):
            audio_chunks.append(chunk)
    
    asyncio.run(run())
    
    # 合并音频
    if audio_chunks:
        audio = np.concatenate(audio_chunks)
    else:
        audio = np.array([])
    
    return audio, inference.get_stats()


if __name__ == "__main__":
    print("=" * 60)
    print("MOSS-Speech Streaming Inference Test")
    print("=" * 60)
    
    # 测试配置
    config = StreamingConfig(
        first_chunk_size=5,
        normal_chunk_size=20,
    )
    
    # 创建推理引擎
    inference = MOSSSpeechStreamingInference(
        engine_path="/workspace/models/MOSS-Speech-TRTLLM-Engine",
        vocoder_path="/workspace/models/BigVGAN",
        config=config,
    )
    
    print(f"Config: first_chunk={config.first_chunk_size}, normal_chunk={config.normal_chunk_size}")
    
    try:
        inference.load()
        
        # 测试流式生成
        async def test():
            text = "Hello, this is a test of streaming speech synthesis."
            async for chunk in inference.generate_streaming(text):
                print(f"  Got chunk: {len(chunk)} samples")
            print(f"\n📊 Stats: {inference.get_stats()}")
        
        asyncio.run(test())
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()



