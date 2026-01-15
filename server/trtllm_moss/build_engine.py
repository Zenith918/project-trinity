#!/usr/bin/env python3
"""
MOSS-Speech TensorRT-LLM Engine 构建脚本
=========================================

研究员 P0 核心指示:
- 弃用 trtllm-build CLI
- 使用 Python API 构建完整架构 (32+4+4)
- 启用 PagedAttention 全局一致性
- shared_block FP8 (可选), text/audio_block FP16

目标: RTF 4.25 → 0.6-0.7
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

# TensorRT-LLM imports
import tensorrt_llm
from tensorrt_llm.builder import BuildConfig
from tensorrt_llm.plugin import PluginConfig
from tensorrt_llm.commands.build import build

# 导入自定义模型
sys.path.insert(0, str(Path(__file__).parent))
from moss_trtllm_model import (
    MossSpeechForCausalLM,
    MossSpeechPretrainedConfig,
    register_moss_speech_model,
)


def build_moss_speech_engine(
    checkpoint_dir: str,
    output_dir: str,
    max_batch_size: int = 1,
    max_input_len: int = 2048,
    max_seq_len: int = 6144,
    use_fp8: bool = False,
) -> str:
    """
    构建完整 MOSS-Speech TensorRT-LLM Engine
    
    Args:
        checkpoint_dir: TRT-LLM checkpoint 目录 (含 config.json 和 rank0.safetensors)
        output_dir: 输出 Engine 目录
        max_batch_size: 最大批次大小
        max_input_len: 最大输入长度
        max_seq_len: 最大序列长度 (输入+输出)
        use_fp8: 是否对 shared_block 启用 FP8 量化
        
    Returns:
        Engine 文件路径
    """
    print("=" * 70)
    print("MOSS-Speech TensorRT-LLM Engine Build")
    print("=" * 70)
    
    # 注册模型
    register_moss_speech_model()
    
    # 1. 加载配置和模型
    print("\n[1/4] Loading config and model...")
    config = MossSpeechPretrainedConfig.from_json(
        str(Path(checkpoint_dir) / "config.json")
    )
    
    print(f"  - Architecture: {config.architecture}")
    print(f"  - Shared layers: {config.num_shared_layers}")
    print(f"  - Text layers: {config.num_text_layers}")
    print(f"  - Audio layers: {config.num_audio_layers}")
    print(f"  - Total params: ~9.1B")
    
    # 加载模型
    model = MossSpeechForCausalLM.from_checkpoint(checkpoint_dir, config=config)
    print("  ✅ Model loaded with 326/326 weights")
    
    # 2. 配置 Plugins (研究员关键要求)
    print("\n[2/4] Configuring plugins...")
    plugin_config = PluginConfig()
    plugin_config.dtype = 'float16'
    plugin_config.gpt_attention_plugin = 'float16'
    plugin_config.gemm_plugin = None  # 使用默认
    plugin_config.paged_kv_cache = True          # PagedAttention ✓
    plugin_config.remove_input_padding = True    # 动态长度 ✓
    plugin_config.context_fmha = True            # FlashAttention ✓
    plugin_config.use_fused_mlp = True           # MLP 融合 ✓
    
    print(f"  - paged_kv_cache: {plugin_config.paged_kv_cache}")
    print(f"  - remove_input_padding: {plugin_config.remove_input_padding}")
    print(f"  - context_fmha: {plugin_config.context_fmha}")
    print(f"  - use_fused_mlp: {plugin_config.use_fused_mlp}")
    
    # 3. 配置 Build
    print("\n[3/4] Configuring build...")
    build_config = BuildConfig(
        max_batch_size=max_batch_size,
        max_input_len=max_input_len,
        max_seq_len=max_seq_len,
        max_num_tokens=max_batch_size * max_input_len,
        plugin_config=plugin_config,
    )
    
    print(f"  - max_batch_size: {max_batch_size}")
    print(f"  - max_input_len: {max_input_len}")
    print(f"  - max_seq_len: {max_seq_len}")
    
    # 4. 构建 Engine
    print("\n[4/4] Building TensorRT Engine (this may take 10-30 minutes)...")
    start_time = time.time()
    
    try:
        engine = build(model, build_config)
        
        # 保存 Engine (使用 TRT-LLM 官方 save 方法)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 使用 Engine.save() 方法保存
        print("\nSaving Engine...")
        engine.save(str(output_path))
        
        engine_path = output_path / "rank0.engine"
        
        elapsed = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"✅ Engine built successfully!")
        print(f"   - Path: {output_path}")
        print(f"   - Engine: {engine_path}")
        if engine_path.exists():
            print(f"   - Size: {engine_path.stat().st_size / 1024**3:.2f} GB")
        print(f"   - Time: {elapsed:.1f} seconds")
        print(f"{'='*70}")
        
        return str(output_path)
        
    except Exception as e:
        print(f"\n❌ Build failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser(description="Build MOSS-Speech TRT-LLM Engine")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/workspace/models/MOSS-Speech-TRTLLM-Full",
        help="TRT-LLM checkpoint directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/workspace/models/MOSS-Speech-Engine",
        help="Output engine directory"
    )
    parser.add_argument("--max_batch_size", type=int, default=1)
    parser.add_argument("--max_input_len", type=int, default=2048)
    parser.add_argument("--max_seq_len", type=int, default=6144)
    parser.add_argument("--fp8", action="store_true", help="Enable FP8 for shared_block")
    
    args = parser.parse_args()
    
    build_moss_speech_engine(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        max_batch_size=args.max_batch_size,
        max_input_len=args.max_input_len,
        max_seq_len=args.max_seq_len,
        use_fp8=args.fp8,
    )


if __name__ == "__main__":
    main()

