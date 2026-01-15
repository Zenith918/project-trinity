"""
MOSS-Speech → TensorRT-LLM 权重转换 (修复版)
=============================================

关键修复:
1. 添加 QKV bias (零初始化，因为 MOSS-Speech 没有 bias)
2. 添加 transformer.ln_f.weight
3. 正确映射所有 Qwen2 期望的权重键名
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_hf_weights(model_path: str, dtype: torch.dtype = torch.float16) -> Dict[str, torch.Tensor]:
    """加载 HuggingFace safetensors 权重"""
    from safetensors import safe_open
    
    weights = {}
    model_path = Path(model_path)
    
    safetensor_files = sorted(model_path.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files in {model_path}")
    
    logger.info(f"Found {len(safetensor_files)} safetensors files")
    
    for sf_file in safetensor_files:
        logger.info(f"Loading {sf_file.name}...")
        with safe_open(sf_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                if tensor.dtype in (torch.float32, torch.float16, torch.bfloat16):
                    tensor = tensor.to(dtype)
                weights[key] = tensor
    
    logger.info(f"Loaded {len(weights)} tensors")
    return weights


def convert_moss_to_qwen2_trtllm(hf_weights: Dict[str, torch.Tensor], num_layers: int = 32) -> Dict[str, torch.Tensor]:
    """
    将 MOSS-Speech 权重转换为 TRT-LLM Qwen2 格式
    
    只转换 shared_block (32层)，忽略 text_block 和 audio_block
    
    MOSS-Speech 键名:
        model.shared_block.layers.{i}.self_attn.q_proj.weight
        model.shared_block.layers.{i}.self_attn.k_proj.weight
        model.shared_block.layers.{i}.self_attn.v_proj.weight
        model.shared_block.layers.{i}.self_attn.o_proj.weight
        model.shared_block.layers.{i}.self_attn.q_norm.weight  (Qwen2 QK Norm)
        model.shared_block.layers.{i}.self_attn.k_norm.weight
        model.shared_block.layers.{i}.mlp.gate_proj.weight
        model.shared_block.layers.{i}.mlp.up_proj.weight
        model.shared_block.layers.{i}.mlp.down_proj.weight
        model.shared_block.layers.{i}.input_layernorm.weight
        model.shared_block.layers.{i}.post_attention_layernorm.weight
    
    TRT-LLM Qwen2 期望:
        transformer.vocab_embedding.weight
        transformer.layers.{i}.attention.qkv.weight  (合并 QKV)
        transformer.layers.{i}.attention.qkv.bias    (需要零初始化)
        transformer.layers.{i}.attention.dense.weight
        transformer.layers.{i}.mlp.gate.weight
        transformer.layers.{i}.mlp.fc.weight
        transformer.layers.{i}.mlp.proj.weight
        transformer.layers.{i}.input_layernorm.weight
        transformer.layers.{i}.post_layernorm.weight
        transformer.ln_f.weight
        lm_head.weight
    """
    trtllm_weights = {}
    
    # 获取模型配置
    # 从第一个权重推断维度
    sample_key = "model.shared_block.layers.0.self_attn.q_proj.weight"
    if sample_key in hf_weights:
        hidden_size = hf_weights[sample_key].shape[1]
        head_dim = hf_weights[sample_key].shape[0] // 32  # num_attention_heads
    else:
        hidden_size = 4096
        head_dim = 128
    
    logger.info(f"Detected hidden_size={hidden_size}, head_dim={head_dim}")
    
    dtype = next(iter(hf_weights.values())).dtype
    
    # === 1. Embeddings ===
    if "model.embed_tokens.weight" in hf_weights:
        trtllm_weights["transformer.vocab_embedding.weight"] = hf_weights["model.embed_tokens.weight"]
        logger.info("✓ vocab_embedding")
    
    # === 2. 转换 32 层 shared_block ===
    for layer_idx in range(num_layers):
        prefix = f"model.shared_block.layers.{layer_idx}"
        trt_prefix = f"transformer.layers.{layer_idx}"
        
        # 2.1 合并 QKV
        q_key = f"{prefix}.self_attn.q_proj.weight"
        k_key = f"{prefix}.self_attn.k_proj.weight"
        v_key = f"{prefix}.self_attn.v_proj.weight"
        
        if q_key in hf_weights and k_key in hf_weights and v_key in hf_weights:
            q = hf_weights[q_key]  # [num_heads * head_dim, hidden]
            k = hf_weights[k_key]  # [num_kv_heads * head_dim, hidden]
            v = hf_weights[v_key]  # [num_kv_heads * head_dim, hidden]
            
            # TRT-LLM 期望 QKV 合并
            qkv = torch.cat([q, k, v], dim=0)
            trtllm_weights[f"{trt_prefix}.attention.qkv.weight"] = qkv
            
            # 添加零 bias (MOSS-Speech 没有 bias，但 TRT-LLM Qwen 需要)
            qkv_bias = torch.zeros(qkv.shape[0], dtype=dtype)
            trtllm_weights[f"{trt_prefix}.attention.qkv.bias"] = qkv_bias
            
            logger.debug(f"✓ Layer {layer_idx} QKV: {qkv.shape}")
        
        # 2.2 Attention Dense (O proj)
        o_key = f"{prefix}.self_attn.o_proj.weight"
        if o_key in hf_weights:
            trtllm_weights[f"{trt_prefix}.attention.dense.weight"] = hf_weights[o_key]
        
        # 2.3 MLP
        # gate_proj -> mlp.gate
        gate_key = f"{prefix}.mlp.gate_proj.weight"
        if gate_key in hf_weights:
            trtllm_weights[f"{trt_prefix}.mlp.gate.weight"] = hf_weights[gate_key]
        
        # up_proj -> mlp.fc
        up_key = f"{prefix}.mlp.up_proj.weight"
        if up_key in hf_weights:
            trtllm_weights[f"{trt_prefix}.mlp.fc.weight"] = hf_weights[up_key]
        
        # down_proj -> mlp.proj
        down_key = f"{prefix}.mlp.down_proj.weight"
        if down_key in hf_weights:
            trtllm_weights[f"{trt_prefix}.mlp.proj.weight"] = hf_weights[down_key]
        
        # 2.4 LayerNorms
        input_ln_key = f"{prefix}.input_layernorm.weight"
        if input_ln_key in hf_weights:
            trtllm_weights[f"{trt_prefix}.input_layernorm.weight"] = hf_weights[input_ln_key]
        
        post_ln_key = f"{prefix}.post_attention_layernorm.weight"
        if post_ln_key in hf_weights:
            trtllm_weights[f"{trt_prefix}.post_layernorm.weight"] = hf_weights[post_ln_key]
    
    # === 3. Final LayerNorm ===
    # MOSS-Speech 有 text_norm 和 audio_norm，我们用 text_norm 作为 ln_f
    if "model.text_norm.weight" in hf_weights:
        trtllm_weights["transformer.ln_f.weight"] = hf_weights["model.text_norm.weight"]
        logger.info("✓ ln_f (from text_norm)")
    else:
        # 创建默认的 ln_f
        ln_f = torch.ones(hidden_size, dtype=dtype)
        trtllm_weights["transformer.ln_f.weight"] = ln_f
        logger.warning("⚠ Created default ln_f.weight")
    
    # === 4. LM Head ===
    if "text_lm_head.weight" in hf_weights:
        trtllm_weights["lm_head.weight"] = hf_weights["text_lm_head.weight"]
        logger.info("✓ lm_head")
    
    logger.info(f"Total TRT-LLM weights: {len(trtllm_weights)}")
    return trtllm_weights


def save_checkpoint(
    weights: Dict[str, torch.Tensor],
    output_dir: str,
    config: dict,
) -> str:
    """保存 TRT-LLM checkpoint"""
    from safetensors.torch import save_file
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存权重
    checkpoint_path = output_path / "rank0.safetensors"
    logger.info(f"Saving checkpoint to {checkpoint_path}")
    save_file(weights, str(checkpoint_path))
    
    # 保存配置
    config_path = output_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"✅ Saved: {checkpoint_path}")
    return str(output_path)


def main():
    """主转换流程"""
    logger.info("=" * 60)
    logger.info("MOSS-Speech → TensorRT-LLM Qwen2 转换 (V2)")
    logger.info("=" * 60)
    
    # 配置
    hf_model_path = "/workspace/models/MOSS-Speech"
    output_dir = "/workspace/models/MOSS-Speech-TRTLLM"
    dtype = torch.float16
    
    # 1. 加载权重
    logger.info(f"Loading from {hf_model_path}")
    hf_weights = load_hf_weights(hf_model_path, dtype)
    
    # 2. 转换
    logger.info("Converting to TRT-LLM Qwen2 format...")
    trtllm_weights = convert_moss_to_qwen2_trtllm(hf_weights, num_layers=32)
    
    # 3. 配置 (Qwen2 兼容)
    config = {
        "architecture": "Qwen2ForCausalLM",
        "dtype": "float16",
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 151680,
        "max_position_embeddings": 40960,
        "quantization": {
            "quant_algo": None,
            "kv_cache_quant_algo": None,
        },
        "mapping": {
            "tp_size": 1,
            "pp_size": 1,
        },
        "head_size": 128,
        "rotary_pct": 1.0,
        "rotary_base": 10000,
        "hidden_act": "silu",
        "qwen_type": "qwen2",
        # Qwen2 特有
        "bias": True,  # 告诉 TRT-LLM 我们有 bias
        "mlp_bias": False,  # MLP 没有 bias
    }
    
    # 4. 保存
    output = save_checkpoint(trtllm_weights, output_dir, config)
    
    logger.info(f"✅ Conversion complete: {output}")
    
    # 5. 验证
    logger.info("\n=== 验证输出 ===")
    for key in sorted(trtllm_weights.keys())[:10]:
        shape = trtllm_weights[key].shape
        logger.info(f"  {key}: {shape}")
    logger.info(f"  ... (共 {len(trtllm_weights)} 个权重)")


if __name__ == "__main__":
    main()



