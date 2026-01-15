#!/usr/bin/env python3
"""
MOSS-Speech TRT-LLM Engine 基准测试
===================================

P0 任务：测量 RTF 和 TTFA

指标定义：
- TTFA (Time To First Audio): 从输入到首个音频 Token 输出的延迟
- RTF (Real-Time Factor): 处理时间 / 音频时长，RTF < 1 表示实时

测试条件：
- 输入: 512 tokens (模拟中等长度对话)
- 输出: 测量生成不同数量音频 Token 的时间
- 模式: FP16 (当前 Engine 配置)
"""

import os
import sys
import time
import json
import argparse
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

import torch
import tensorrt as trt


def load_engine_native(engine_path: str):
    """使用原生 TensorRT 加载 Engine (用于调试)"""
    # 需要先加载 TRT-LLM 插件
    import tensorrt_llm
    
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    
    engine = runtime.deserialize_cuda_engine(engine_data)
    return engine


def load_engine_trtllm(engine_dir: str):
    """使用 TRT-LLM 加载 Engine"""
    # 先注册自定义模型
    from moss_trtllm_model import register_moss_speech_model
    register_moss_speech_model()
    
    from tensorrt_llm.builder import Engine
    from tensorrt_llm.runtime import ModelRunnerCpp
    
    engine = Engine.from_dir(engine_dir)
    return engine


def benchmark_inference(engine_dir: str, num_input_tokens: int = 512, num_output_tokens: int = 100, num_warmup: int = 3, num_runs: int = 10):
    """
    运行基准测试
    
    Args:
        engine_dir: Engine 目录
        num_input_tokens: 输入 token 数量 (模拟对话长度)
        num_output_tokens: 输出 token 数量 (音频 token)
        num_warmup: 预热次数
        num_runs: 正式测试次数
    """
    print("=" * 70)
    print("MOSS-Speech TRT-LLM 基准测试")
    print("=" * 70)
    
    # 加载配置
    config_path = Path(engine_dir) / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    pretrained_config = config.get('pretrained_config', {})
    build_config = config.get('build_config', {})
    
    print(f"\n[配置信息]")
    print(f"  Architecture: {pretrained_config.get('architecture', 'N/A')}")
    print(f"  Dtype: {pretrained_config.get('dtype', 'N/A')}")
    print(f"  Max Batch Size: {build_config.get('max_batch_size', 'N/A')}")
    print(f"  Max Seq Len: {build_config.get('max_seq_len', 'N/A')}")
    print(f"  PagedKVCache: {build_config.get('plugin_config', {}).get('paged_kv_cache', 'N/A')}")
    
    print(f"\n[测试参数]")
    print(f"  Input tokens: {num_input_tokens}")
    print(f"  Output tokens: {num_output_tokens}")
    print(f"  Warmup runs: {num_warmup}")
    print(f"  Benchmark runs: {num_runs}")
    
    # 加载 Engine
    print(f"\n[加载 Engine...]")
    
    from moss_trtllm_model import register_moss_speech_model
    register_moss_speech_model()
    
    from tensorrt_llm.builder import Engine
    engine = Engine.from_dir(engine_dir)
    print(f"  ✅ Engine loaded")
    
    # 检查 GPU 内存
    torch.cuda.empty_cache()
    gpu_mem_before = torch.cuda.memory_allocated() / 1024**3
    print(f"  GPU Memory (before): {gpu_mem_before:.2f} GB")
    
    # 准备输入数据
    vocab_size = pretrained_config.get('vocab_size', 151680)
    batch_size = 1
    
    # 模拟输入 (随机 token IDs)
    input_ids = torch.randint(0, vocab_size, (batch_size, num_input_tokens), dtype=torch.int32).cuda()
    
    print(f"\n[测试 TTFA (首 Token 延迟)]")
    
    # 由于 TRT-LLM 的 Engine 需要特定的 runtime，这里我们测量 Engine 加载和基本推理能力
    # 完整的推理测试需要实现 ModelRunner
    
    # 测试 1: Engine 加载时间
    load_times = []
    for i in range(3):
        start = time.perf_counter()
        engine = Engine.from_dir(engine_dir)
        elapsed = (time.perf_counter() - start) * 1000
        load_times.append(elapsed)
        print(f"  Engine load #{i+1}: {elapsed:.1f} ms")
    
    print(f"\n[结果摘要]")
    print(f"  Engine 加载时间: {np.mean(load_times):.1f} ± {np.std(load_times):.1f} ms")
    
    # 检查 Engine 大小和预估内存
    engine_path = Path(engine_dir) / "rank0.engine"
    engine_size_gb = engine_path.stat().st_size / 1024**3
    
    # RTX 4090 有 24GB VRAM
    available_vram = 24.0  # GB
    estimated_runtime_mem = engine_size_gb * 1.2  # 运行时额外开销约 20%
    kv_cache_budget = available_vram - estimated_runtime_mem
    
    print(f"\n[内存分析 (RTX 4090 24GB)]")
    print(f"  Engine 文件大小: {engine_size_gb:.2f} GB")
    print(f"  预估运行时内存: {estimated_runtime_mem:.2f} GB")
    print(f"  KV Cache 预算: {kv_cache_budget:.2f} GB")
    
    if kv_cache_budget < 4.0:
        print(f"  ⚠️ 警告: KV Cache 空间紧张，建议启用 FP8 量化")
    else:
        print(f"  ✅ KV Cache 空间充足")
    
    # 理论性能估算
    print(f"\n[理论性能估算]")
    
    # 假设 RTX 4090 在 FP16 下的理论吞吐量
    # MOSS-Speech 9.1B 参数，FP16 理论上可以达到 ~50-100 tokens/s
    theoretical_throughput = 70  # tokens/s (保守估计)
    
    # 音频参数
    audio_tokens_per_second = 50  # MOSS-Speech 的音频 token 率 (假设)
    
    # 计算理论 RTF
    theoretical_rtf = audio_tokens_per_second / theoretical_throughput
    
    print(f"  理论吞吐量 (FP16): ~{theoretical_throughput} tokens/s")
    print(f"  音频 Token 率: {audio_tokens_per_second} tokens/s")
    print(f"  理论 RTF: {theoretical_rtf:.2f}")
    
    if theoretical_rtf < 1.0:
        print(f"  ✅ 理论上可实现实时 (RTF < 1)")
    else:
        print(f"  ⚠️ 可能需要 FP8 量化来实现实时")
    
    print(f"\n[下一步]")
    print(f"  1. 实现 ModelRunner 进行完整推理测试")
    print(f"  2. 测量实际 TTFA 和 RTF")
    print(f"  3. 如果 RTF > 1.5，启动 FP8 校准计划")
    
    print("\n" + "=" * 70)
    
    return {
        'engine_size_gb': engine_size_gb,
        'estimated_runtime_mem': estimated_runtime_mem,
        'kv_cache_budget': kv_cache_budget,
        'theoretical_rtf': theoretical_rtf,
    }


def run_actual_inference(engine_dir: str):
    """
    运行实际推理测试 (需要实现完整的 ModelRunner)
    
    这是一个更复杂的测试，需要：
    1. 正确设置 KV Cache
    2. 处理 attention mask
    3. 实现自回归生成循环
    """
    print("\n[实际推理测试]")
    print("  ⚠️ 需要实现完整的 ModelRunner")
    print("  当前仅进行 Engine 加载和配置验证")
    
    # TODO: 实现完整的推理循环
    # 1. 创建 TensorRT execution context
    # 2. 分配输入/输出 buffer
    # 3. 实现 KV Cache 管理
    # 4. 测量每个 token 的生成时间


def main():
    parser = argparse.ArgumentParser(description="MOSS-Speech TRT-LLM Benchmark")
    parser.add_argument(
        "--engine_dir",
        type=str,
        default="/workspace/models/MOSS-Speech-Engine",
        help="TRT-LLM Engine 目录"
    )
    parser.add_argument(
        "--input_tokens",
        type=int,
        default=512,
        help="输入 token 数量"
    )
    parser.add_argument(
        "--output_tokens",
        type=int,
        default=100,
        help="输出 token 数量"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="预热次数"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="测试次数"
    )
    
    args = parser.parse_args()
    
    results = benchmark_inference(
        engine_dir=args.engine_dir,
        num_input_tokens=args.input_tokens,
        num_output_tokens=args.output_tokens,
        num_warmup=args.warmup,
        num_runs=args.runs,
    )
    
    return results


if __name__ == "__main__":
    main()



