#!/usr/bin/env python3
"""
Build TensorRT-LLM Engine for MOSS-Speech
==========================================

⚠️ 警告: 本文件包含关键的架构约束，修改前请阅读:
   /workspace/docs/moss-speech/PITFALLS_AND_SOLUTIONS.md

关键参数 (禁止修改):
-------------------
- TOTAL_LAYERS = 40
- max_seq_len = 2048 (扩展需谨慎，见文档)
"""

import os
import sys
import json
import inspect

# ═══════════════════════════════════════════════════════════════════════════════
# 🔴 关键常量 - 禁止修改！
# ═══════════════════════════════════════════════════════════════════════════════
TOTAL_LAYERS = 40  # 32 shared + 4 text + 4 audio

# 默认 max_seq_len = 2048
# ⚠️ 扩展到 4096 需要:
#    1. 启用 FP8 量化
#    2. 或使用多卡分流
#    3. 或确保系统有 > 300GB RAM
DEFAULT_MAX_SEQ_LEN = 2048


def verify_build_prerequisites():
    """
    [ARCH_GUARD] 验证构建前置条件
    """
    assert TOTAL_LAYERS == 40, \
        f"[ARCH_GUARD] FATAL: TOTAL_LAYERS={TOTAL_LAYERS}, 必须为 40！"
    
    print("[ARCH_GUARD] ✅ 构建前置条件验证通过")


def build_moss_speech_engine(
    checkpoint_dir: str,
    output_dir: str,
    max_batch_size: int = 1,
    max_input_len: int = 1024,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
):
    """
    Build TensorRT-LLM engine for MOSS-Speech
    
    Parameters
    ----------
    checkpoint_dir : str
        Checkpoint 目录，包含 config.json 和 rank0.safetensors
    output_dir : str
        输出目录
    max_batch_size : int
        最大 batch size
    max_input_len : int
        最大输入长度
    max_seq_len : int
        最大序列长度（包括生成的 token）
        ⚠️ 当前默认 2048，扩展需谨慎
    """
    import psutil
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [ARCH_GUARD] 构建前验证
    # ═══════════════════════════════════════════════════════════════════════════
    verify_build_prerequisites()
    
    print(f"=" * 60)
    print(f"[ARCH_GUARD] Building MOSS-Speech Engine")
    print(f"[ARCH_GUARD] 40-Layer Virtual Linearization Active")
    print(f"=" * 60)
    print(f"  Checkpoint: {checkpoint_dir}")
    print(f"  Output: {output_dir}")
    print(f"  max_batch_size: {max_batch_size}")
    print(f"  max_input_len: {max_input_len}")
    print(f"  max_seq_len: {max_seq_len}")
    print(f"  TOTAL_LAYERS: {TOTAL_LAYERS}")
    print(f"=" * 60)
    
    # 检查初始内存
    mem = psutil.virtual_memory()
    print(f"初始内存: {mem.percent:.1f}% ({mem.used/1024**3:.1f}GB/{mem.total/1024**3:.1f}GB)")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 导入 TensorRT-LLM
    # ═══════════════════════════════════════════════════════════════════════════
    import tensorrt_llm
    from tensorrt_llm.builder import BuildConfig
    from tensorrt_llm.plugin import PluginConfig
    
    # 导入模型定义
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from moss_trtllm_model import MossSpeechForCausalLM, verify_model_architecture
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 加载配置
    # ═══════════════════════════════════════════════════════════════════════════
    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # 兼容两种配置格式
    if 'pretrained_config' in config_dict:
        config_dict['pretrained_config']['num_hidden_layers'] = TOTAL_LAYERS
        pretrained_config = config_dict['pretrained_config']
    else:
        config_dict['num_hidden_layers'] = TOTAL_LAYERS
        pretrained_config = config_dict
    
    print(f"[ARCH_GUARD] ✅ 强制 num_hidden_layers = {TOTAL_LAYERS}")
    
    # 保存修改后的配置
    modified_config_path = os.path.join(output_dir, "config.json")
    with open(modified_config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"✅ 保存修改后的配置到: {modified_config_path}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 加载模型
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n加载模型...")
    model = MossSpeechForCausalLM.from_checkpoint(checkpoint_dir)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [ARCH_GUARD] 验证模型架构
    # ═══════════════════════════════════════════════════════════════════════════
    verify_model_architecture(model)
    
    # 额外的静态断言
    assert model.config.num_hidden_layers == 40, \
        "[ARCH_GUARD] FATAL: 物理层数必须为40，严禁修改为Qwen默认的32层！"
    
    # 检查虚拟线性化依赖链
    source = inspect.getsource(model.transformer.forward)
    assert "1e-4" in source or "1e-04" in source or "VIRTUAL_LINEARIZATION_EPSILON" in source, \
        "[ARCH_GUARD] FATAL: 虚拟序列化依赖链丢失，Generation Phase 将崩溃！"
    
    # 检查内存
    mem = psutil.virtual_memory()
    print(f"模型加载后内存: {mem.percent:.1f}%")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 配置插件
    # ═══════════════════════════════════════════════════════════════════════════
    plugin_config = PluginConfig()
    plugin_config.gpt_attention_plugin = "float16"
    plugin_config.gemm_plugin = "float16"
    plugin_config.paged_kv_cache = True   # 🔴 必须启用 PagedAttention
    plugin_config.remove_input_padding = True
    plugin_config.context_fmha = True
    
    # 计算 KV cache 块数
    tokens_per_block = 64
    max_blocks_per_seq = (max_seq_len + tokens_per_block - 1) // tokens_per_block
    print(f"max_blocks_per_seq: {max_blocks_per_seq} (tokens_per_block={tokens_per_block})")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 构建配置
    # ═══════════════════════════════════════════════════════════════════════════
    build_config = BuildConfig(
        max_batch_size=max_batch_size,
        max_input_len=max_input_len,
        max_seq_len=max_seq_len,
        max_num_tokens=max_batch_size * max_input_len,
        plugin_config=plugin_config,
        strongly_typed=True,  # 启用强类型以减少内存
    )
    
    print(f"[ARCH_GUARD] ✅ 模型层数已设置为 {TOTAL_LAYERS}")
    
    # 检查内存
    mem = psutil.virtual_memory()
    print(f"构建配置后内存: {mem.percent:.1f}%")
    
    if mem.percent > 80:
        print(f"⚠️ 内存较高 ({mem.percent:.1f}%)，继续构建...")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 构建 Engine
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n开始构建 Engine...")
    print(f"这可能需要 10-30 分钟，请耐心等待...")
    
    try:
        engine = tensorrt_llm.builder.build(model, build_config)
        
        # 保存 Engine
        engine_path = os.path.join(output_dir, "rank0.engine")
        print(f"\n保存 Engine 到: {engine_path}")
        
        with open(engine_path, 'wb') as f:
            f.write(engine.engine)
        
        # ═══════════════════════════════════════════════════════════════════════
        # 保存配置（包含架构信息）
        # ═══════════════════════════════════════════════════════════════════════
        engine_config = {
            "pretrained_config": pretrained_config,
            "build_config": {
                "max_batch_size": max_batch_size,
                "max_input_len": max_input_len,
                "max_seq_len": max_seq_len,
                "num_layers": TOTAL_LAYERS,
                "paged_kv_cache": True,
                "tokens_per_block": tokens_per_block,
                "max_blocks_per_seq": max_blocks_per_seq,
            },
            # [ARCH_GUARD] 记录架构信息
            "arch_guard": {
                "virtual_linearization": True,
                "epsilon": "1e-4",
                "total_layers": TOTAL_LAYERS,
                "audio_start_idx": 36,
            }
        }
        
        engine_config_path = os.path.join(output_dir, "config.json")
        with open(engine_config_path, 'w') as f:
            json.dump(engine_config, f, indent=2)
        
        print(f"\n" + "=" * 60)
        print(f"[ARCH_GUARD] ✅✅✅ Engine built successfully! ✅✅✅")
        print(f"[ARCH_GUARD] 40-Layer Virtual Linearization Active")
        print(f"=" * 60)
        print(f"Engine: {engine_path}")
        print(f"Config: {engine_config_path}")
        
        # 最终内存
        mem = psutil.virtual_memory()
        print(f"最终内存: {mem.percent:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 构建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build MOSS-Speech TensorRT-LLM Engine")
    parser.add_argument("--checkpoint_dir", default="/workspace/models/MOSS-Speech-TRTLLM-Full",
                        help="Checkpoint 目录")
    parser.add_argument("--output_dir", default="/workspace/models/MOSS-Speech-Engine-v9",
                        help="输出目录")
    parser.add_argument("--max_batch_size", type=int, default=1,
                        help="最大 batch size")
    parser.add_argument("--max_input_len", type=int, default=1024,
                        help="最大输入长度")
    parser.add_argument("--max_seq_len", type=int, default=DEFAULT_MAX_SEQ_LEN,
                        help=f"最大序列长度 (默认: {DEFAULT_MAX_SEQ_LEN})")
    args = parser.parse_args()
    
    # 打印警告
    if args.max_seq_len > DEFAULT_MAX_SEQ_LEN:
        print(f"⚠️ 警告: max_seq_len={args.max_seq_len} > {DEFAULT_MAX_SEQ_LEN}")
        print(f"   扩展 max_seq_len 可能导致 OOM！")
        print(f"   请确保:")
        print(f"     1. 系统有足够内存 (> 300GB)")
        print(f"     2. 或启用 FP8 量化")
        print(f"     3. 或使用多卡分流")
        print()
    
    build_moss_speech_engine(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        max_batch_size=args.max_batch_size,
        max_input_len=args.max_input_len,
        max_seq_len=args.max_seq_len,
    )
