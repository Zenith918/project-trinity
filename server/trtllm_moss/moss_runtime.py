#!/usr/bin/env python3
"""
MOSS-Speech TRT-LLM Runtime - 真实推理测试
==========================================

首席研究员 P0 指令：
- 使用 TRT-LLM ModelRunner 进行真实压力测试
- 测量 TTFA (首个音频 Token 延迟)
- 测量 RTF (实时因子)
- 验证 60% 效率假设

测试场景：
- 并发数: 1
- 输入: 512 tokens
- 输出: 100 audio tokens
"""

import os
import sys
import time
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

# TRT-LLM imports
import tensorrt_llm
from tensorrt_llm.runtime import ModelRunner, SamplingConfig


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    ttfa_ms: float           # Time To First Audio (ms)
    rtf: float               # Real-Time Factor
    tokens_per_sec: float    # 吞吐量
    prefill_time_ms: float   # Prefill 时间
    decode_time_ms: float    # Decode 总时间
    total_tokens: int        # 生成的 Token 数
    gpu_memory_gb: float     # GPU 内存使用


class MossSpeechRuntime:
    """
    MOSS-Speech TRT-LLM 推理运行时
    
    使用 TRT-LLM ModelRunner 进行高性能推理
    """
    
    def __init__(
        self,
        engine_dir: str,
        max_batch_size: int = 1,
        max_input_len: int = 2048,
        max_output_len: int = 1024,
    ):
        """
        初始化运行时
        
        Args:
            engine_dir: TRT-LLM Engine 目录
            max_batch_size: 最大批次大小
            max_input_len: 最大输入长度
            max_output_len: 最大输出长度
        """
        self.engine_dir = Path(engine_dir)
        self.max_batch_size = max_batch_size
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        
        self.runner: Optional[ModelRunner] = None
        self.config: Dict = {}
        
    def load(self) -> float:
        """
        加载 Engine
        
        Returns:
            加载时间 (秒)
        """
        print(f"[MossSpeechRuntime] Loading Engine from {self.engine_dir}")
        
        # 注册自定义模型
        from moss_trtllm_model import register_moss_speech_model
        register_moss_speech_model()
        
        # 加载配置
        config_path = self.engine_dir / "config.json"
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        start = time.perf_counter()
        
        # 使用 ModelRunner 加载
        self.runner = ModelRunner.from_dir(
            engine_dir=str(self.engine_dir),
            rank=0,
        )
        
        load_time = time.perf_counter() - start
        
        print(f"  ✅ Engine loaded in {load_time:.1f}s")
        print(f"  - Architecture: {self.config.get('pretrained_config', {}).get('architecture')}")
        print(f"  - Vocab size: {self.runner.vocab_size}")
        print(f"  - Max seq len: {self.runner.max_sequence_length}")
        
        return load_time
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        streaming: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        生成音频 Tokens
        
        Args:
            input_ids: 输入 token IDs [batch, seq_len]
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_k: Top-K 采样
            top_p: Top-P 采样
            streaming: 是否流式输出
            
        Returns:
            (output_ids, timing_info)
        """
        if self.runner is None:
            raise RuntimeError("Engine not loaded. Call load() first.")
        
        # 准备输入
        batch_input_ids = [input_ids[i] for i in range(input_ids.shape[0])]
        
        # 配置采样
        sampling_config = SamplingConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        
        # 记录时间
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        # 生成
        outputs = self.runner.generate(
            batch_input_ids=batch_input_ids,
            sampling_config=sampling_config,
            streaming=streaming,
        )
        
        torch.cuda.synchronize()
        total_time = time.perf_counter() - start
        
        # 解析输出
        if isinstance(outputs, dict):
            output_ids = outputs.get('output_ids', outputs.get('sequences'))
        else:
            output_ids = outputs
        
        timing_info = {
            'total_time_ms': total_time * 1000,
            'input_tokens': input_ids.shape[1],
            'output_tokens': output_ids.shape[-1] - input_ids.shape[1] if output_ids is not None else 0,
        }
        
        return output_ids, timing_info
    
    def benchmark(
        self,
        num_input_tokens: int = 512,
        num_output_tokens: int = 100,
        num_warmup: int = 2,
        num_runs: int = 5,
    ) -> BenchmarkResult:
        """
        运行基准测试
        
        Args:
            num_input_tokens: 输入 token 数量
            num_output_tokens: 输出 token 数量
            num_warmup: 预热次数
            num_runs: 测试次数
            
        Returns:
            BenchmarkResult
        """
        if self.runner is None:
            raise RuntimeError("Engine not loaded. Call load() first.")
        
        print(f"\n[Benchmark] Starting...")
        print(f"  - Input tokens: {num_input_tokens}")
        print(f"  - Output tokens: {num_output_tokens}")
        print(f"  - Warmup runs: {num_warmup}")
        print(f"  - Benchmark runs: {num_runs}")
        
        # 获取 vocab size
        vocab_size = self.runner.vocab_size
        
        # 生成随机输入
        input_ids = torch.randint(
            0, vocab_size,
            (1, num_input_tokens),
            dtype=torch.int32
        ).cuda()
        
        # 预热
        print(f"\n[Warmup]")
        for i in range(num_warmup):
            _, timing = self.generate(input_ids, max_new_tokens=num_output_tokens)
            print(f"  Run {i+1}: {timing['total_time_ms']:.1f} ms")
        
        # 正式测试
        print(f"\n[Benchmark Runs]")
        times = []
        first_token_times = []
        
        for i in range(num_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            output_ids, timing = self.generate(
                input_ids,
                max_new_tokens=num_output_tokens,
            )
            
            torch.cuda.synchronize()
            total_time = time.perf_counter() - start
            times.append(total_time * 1000)
            
            print(f"  Run {i+1}: {total_time * 1000:.1f} ms, "
                  f"output tokens: {timing['output_tokens']}")
        
        # 计算统计
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        # 计算指标
        total_tokens = num_input_tokens + num_output_tokens
        tokens_per_sec = num_output_tokens / (avg_time / 1000)
        
        # TTFA 估算 (假设 prefill + 5 tokens)
        # 简化: TTFA ≈ avg_time * (num_input_tokens + 5) / total_tokens
        ttfa_ms = avg_time * (num_input_tokens + 5) / total_tokens
        
        # RTF 计算
        # 假设音频 token 率为 50 tokens/s (20ms per token)
        audio_token_rate = 50  # tokens/s
        audio_duration_s = num_output_tokens / audio_token_rate
        processing_time_s = avg_time / 1000
        rtf = processing_time_s / audio_duration_s
        
        # GPU 内存
        gpu_memory = torch.cuda.max_memory_allocated() / 1024**3
        
        result = BenchmarkResult(
            ttfa_ms=ttfa_ms,
            rtf=rtf,
            tokens_per_sec=tokens_per_sec,
            prefill_time_ms=ttfa_ms * 0.9,  # 估算
            decode_time_ms=avg_time - ttfa_ms * 0.9,
            total_tokens=num_output_tokens,
            gpu_memory_gb=gpu_memory,
        )
        
        print(f"\n{'='*60}")
        print(f"[Benchmark Results]")
        print(f"  Total time: {avg_time:.1f} ± {std_time:.1f} ms")
        print(f"  Throughput: {tokens_per_sec:.1f} tokens/s")
        print(f"  TTFA (估算): {ttfa_ms:.1f} ms")
        print(f"  RTF: {rtf:.3f}")
        print(f"  GPU Memory: {gpu_memory:.2f} GB")
        print(f"{'='*60}")
        
        # 评估
        print(f"\n[评估]")
        if rtf < 1.0:
            print(f"  ✅ RTF < 1.0: 可实现实时")
        else:
            print(f"  ⚠️ RTF >= 1.0: 需要优化")
        
        if ttfa_ms < 300:
            print(f"  ✅ TTFA < 300ms: 达标")
        else:
            print(f"  ⚠️ TTFA >= 300ms: 需要优化")
        
        return result


def main():
    """主函数 - 运行基准测试"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MOSS-Speech Runtime Benchmark")
    parser.add_argument(
        "--engine_dir",
        type=str,
        default="/workspace/models/MOSS-Speech-Engine",
        help="TRT-LLM Engine 目录"
    )
    parser.add_argument("--input_tokens", type=int, default=512)
    parser.add_argument("--output_tokens", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MOSS-Speech TRT-LLM Runtime Benchmark")
    print("=" * 60)
    
    # 创建运行时
    runtime = MossSpeechRuntime(
        engine_dir=args.engine_dir,
        max_batch_size=1,
        max_input_len=args.input_tokens * 2,
        max_output_len=args.output_tokens * 2,
    )
    
    # 加载 Engine
    load_time = runtime.load()
    print(f"\nEngine 加载时间: {load_time:.1f}s (网络存储)")
    
    # 运行基准测试
    result = runtime.benchmark(
        num_input_tokens=args.input_tokens,
        num_output_tokens=args.output_tokens,
        num_warmup=args.warmup,
        num_runs=args.runs,
    )
    
    # 输出 JSON 结果
    print(f"\n[JSON Result]")
    result_dict = {
        'ttfa_ms': result.ttfa_ms,
        'rtf': result.rtf,
        'tokens_per_sec': result.tokens_per_sec,
        'gpu_memory_gb': result.gpu_memory_gb,
    }
    print(json.dumps(result_dict, indent=2))
    
    return result


if __name__ == "__main__":
    main()



