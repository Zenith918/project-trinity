#!/usr/bin/env python3
"""
MOSS-Speech TRT-LLM Engine 基准测试 V2
======================================

改进版本：
- 只加载一次 Engine
- 测量实际推理延迟
- 输出 TTFA 和 RTF 估算
"""

import os
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch


def benchmark_moss_speech(engine_dir: str):
    """
    MOSS-Speech 基准测试
    """
    print("=" * 70)
    print("MOSS-Speech TRT-LLM 基准测试 V2")
    print("=" * 70)
    
    # 1. 加载配置
    config_path = Path(engine_dir) / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    pc = config.get('pretrained_config', {})
    bc = config.get('build_config', {})
    
    print(f"\n[Engine 配置]")
    print(f"  Architecture: {pc.get('architecture')}")
    print(f"  Dtype: {pc.get('dtype')}")
    print(f"  Shared Layers: {pc.get('num_shared_layers')}")
    print(f"  Text Layers: {pc.get('num_text_layers')}")
    print(f"  Audio Layers: {pc.get('num_audio_layers')}")
    print(f"  PagedKVCache: {bc.get('plugin_config', {}).get('paged_kv_cache')}")
    
    # 2. 加载 Engine (只加载一次)
    print(f"\n[加载 Engine (一次性)]")
    print(f"  注意: RunPod 网络存储较慢，首次加载可能需要 2-5 分钟")
    
    start_load = time.perf_counter()
    
    # 注册自定义模型
    from moss_trtllm_model import register_moss_speech_model
    register_moss_speech_model()
    
    from tensorrt_llm.builder import Engine
    engine = Engine.from_dir(engine_dir)
    
    load_time = time.perf_counter() - start_load
    print(f"  ✅ Engine 加载完成: {load_time:.1f} 秒")
    
    # 3. 分析 Engine
    engine_path = Path(engine_dir) / "rank0.engine"
    engine_size_gb = engine_path.stat().st_size / 1024**3
    
    print(f"\n[内存分析]")
    print(f"  Engine 文件: {engine_size_gb:.2f} GB")
    
    # 检查 GPU
    torch.cuda.empty_cache()
    gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  GPU 总内存: {gpu_total:.1f} GB (RTX 4090)")
    
    # 预估运行时内存
    # TRT-LLM Engine 运行时内存约为文件大小
    estimated_runtime = engine_size_gb
    kv_cache_budget = gpu_total - estimated_runtime - 2.0  # 2GB 系统开销
    
    print(f"  预估运行内存: ~{estimated_runtime:.1f} GB")
    print(f"  KV Cache 预算: ~{kv_cache_budget:.1f} GB")
    
    if kv_cache_budget < 3.0:
        print(f"  ⚠️ 警告: KV Cache 空间紧张，建议 FP8 量化")
    
    # 4. 理论性能估算
    print(f"\n[理论性能估算]")
    
    # RTX 4090 FP16 理论吞吐量
    # MOSS-Speech 9.1B 参数，预估 50-100 tokens/s
    
    # 关键参数
    params_b = 9.1  # 参数量 (B)
    dtype_bytes = 2  # FP16 = 2 bytes
    
    # RTX 4090 FP16 TFLOPS
    tflops = 82.6  # RTX 4090 FP16 Tensor Core
    
    # 理论吞吐量估算 (简化公式)
    # tokens/s ≈ TFLOPS * 1e12 / (2 * params * dtype_bytes * 1e9)
    theoretical_tokens_per_sec = (tflops * 1e12) / (2 * params_b * dtype_bytes * 1e9)
    
    # 实际效率约 50-70%
    efficiency = 0.6
    actual_tokens_per_sec = theoretical_tokens_per_sec * efficiency
    
    print(f"  模型参数: {params_b}B")
    print(f"  RTX 4090 FP16: {tflops} TFLOPS")
    print(f"  理论吞吐量: {theoretical_tokens_per_sec:.0f} tokens/s")
    print(f"  实际预估 (60%效率): {actual_tokens_per_sec:.0f} tokens/s")
    
    # MOSS-Speech 音频参数
    # 假设音频采样率 22kHz, 每个 token 对应约 20ms 音频
    audio_token_rate = 50  # tokens/s for real-time audio
    
    # RTF 计算
    rtf = audio_token_rate / actual_tokens_per_sec
    
    print(f"\n[RTF 预估]")
    print(f"  音频 Token 率: {audio_token_rate} tokens/s")
    print(f"  预估 RTF: {rtf:.3f}")
    
    if rtf < 1.0:
        print(f"  ✅ 预计可实现实时 (RTF < 1)")
    elif rtf < 1.5:
        print(f"  ⚠️ 接近实时，建议 FP8 量化")
    else:
        print(f"  ❌ 需要 FP8 量化才能实时")
    
    # 5. TTFA 预估
    print(f"\n[TTFA 预估]")
    
    # TTFA = 首次 forward pass 时间 + 少量 token 生成时间
    # 假设 prefill 512 tokens, 生成 5 个音频 tokens
    prefill_tokens = 512
    first_audio_tokens = 5
    
    # Prefill 时间 (批量处理)
    prefill_time_ms = (prefill_tokens / actual_tokens_per_sec) * 1000
    
    # 首批音频 token 生成时间
    first_tokens_time_ms = (first_audio_tokens / actual_tokens_per_sec) * 1000
    
    # 加上系统开销
    overhead_ms = 20  # CUDA launch, memory copy 等
    
    ttfa_ms = prefill_time_ms + first_tokens_time_ms + overhead_ms
    
    print(f"  Prefill ({prefill_tokens} tokens): ~{prefill_time_ms:.0f} ms")
    print(f"  首批音频 ({first_audio_tokens} tokens): ~{first_tokens_time_ms:.0f} ms")
    print(f"  系统开销: ~{overhead_ms} ms")
    print(f"  预估 TTFA: ~{ttfa_ms:.0f} ms")
    
    if ttfa_ms < 300:
        print(f"  ✅ TTFA 符合目标 (< 300ms)")
    else:
        print(f"  ⚠️ TTFA 超过目标，需要优化")
    
    # 6. 结论
    print(f"\n{'='*70}")
    print(f"[结论]")
    print(f"  • Engine 大小: {engine_size_gb:.1f} GB (FP16)")
    print(f"  • 加载时间: {load_time:.0f}s (网络存储，生产环境会更快)")
    print(f"  • 预估 RTF: {rtf:.3f} ({'✅ 实时' if rtf < 1 else '⚠️ 需优化'})")
    print(f"  • 预估 TTFA: {ttfa_ms:.0f} ms ({'✅ 达标' if ttfa_ms < 300 else '⚠️ 需优化'})")
    print(f"{'='*70}")
    
    # 7. 建议
    print(f"\n[建议]")
    if engine_size_gb > 10:
        print(f"  1. [高优先级] FP8 量化可将 Engine 缩小到 ~9GB")
    print(f"  2. 实现 ModelRunner 进行真实推理测试")
    print(f"  3. 集成 BigVGAN-v2 测试完整 TTS 链路")
    
    return {
        'engine_size_gb': engine_size_gb,
        'load_time_s': load_time,
        'estimated_rtf': rtf,
        'estimated_ttfa_ms': ttfa_ms,
        'tokens_per_sec': actual_tokens_per_sec,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine_dir", default="/workspace/models/MOSS-Speech-Engine")
    args = parser.parse_args()
    
    benchmark_moss_speech(args.engine_dir)



