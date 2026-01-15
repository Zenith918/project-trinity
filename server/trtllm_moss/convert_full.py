"""
MOSS-Speech 完整权重转换 (32+4+4 架构)
======================================

研究员 P0 指示: 包含 shared_block + text_block + audio_block

输出结构:
- shared_block.layers.{0-31}.*
- text_block.layers.{0-3}.*
- audio_block.layers.{0-3}.*
- text_norm, audio_norm
- text_lm_head, audio_lm_head
- embed_tokens, audio_embed
"""

import os
import json
import torch
from pathlib import Path
from typing import Dict
from safetensors import safe_open
from safetensors.torch import save_file
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_moss_weights(model_path: str, dtype: torch.dtype = torch.float16) -> Dict[str, torch.Tensor]:
    """加载 MOSS-Speech HuggingFace 权重"""
    weights = {}
    model_path = Path(model_path)
    
    safetensor_files = sorted(model_path.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors in {model_path}")
    
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


def convert_full_moss_speech(
    hf_weights: Dict[str, torch.Tensor],
    dtype: torch.dtype = torch.float16,
) -> Dict[str, torch.Tensor]:
    """
    转换完整 MOSS-Speech 权重
    
    HuggingFace 键名:
        model.shared_block.layers.{i}.self_attn.{q,k,v,o}_proj.weight
        model.shared_block.layers.{i}.self_attn.{q,k}_norm.weight
        model.shared_block.layers.{i}.mlp.{gate,up,down}_proj.weight
        model.shared_block.layers.{i}.{input,post_attention}_layernorm.weight
        model.text_block.layers.{i}.*
        model.audio_block.layers.{i}.*
        model.{text,audio}_norm.weight
        {text,audio}_lm_head.weight
        model.embed_tokens.weight
        model.audio_embed.weight
    
    TRT-LLM 输出键名:
        shared_block.layers.{i}.attention.qkv.weight
        shared_block.layers.{i}.attention.qkv.bias
        shared_block.layers.{i}.attention.dense.weight
        shared_block.layers.{i}.mlp.gate.weight
        shared_block.layers.{i}.mlp.fc.weight
        shared_block.layers.{i}.mlp.proj.weight
        shared_block.layers.{i}.input_layernorm.weight
        shared_block.layers.{i}.post_layernorm.weight
        text_block.layers.{i}.*
        audio_block.layers.{i}.*
        text_norm.weight
        audio_norm.weight
        text_lm_head.weight
        audio_lm_head.weight
        embed_tokens.weight
        audio_embed.weight
    """
    trtllm_weights = {}
    
    # === 1. 嵌入层 ===
    if "model.embed_tokens.weight" in hf_weights:
        trtllm_weights["embed_tokens.weight"] = hf_weights["model.embed_tokens.weight"]
        logger.info("✓ embed_tokens")
    
    if "model.audio_embed.weight" in hf_weights:
        trtllm_weights["audio_embed.weight"] = hf_weights["model.audio_embed.weight"]
        logger.info("✓ audio_embed")
    
    # === 2. Shared Block (32 层) ===
    logger.info("Converting shared_block (32 layers)...")
    for layer_idx in range(32):
        _convert_decoder_layer(
            hf_weights=hf_weights,
            trtllm_weights=trtllm_weights,
            hf_prefix=f"model.shared_block.layers.{layer_idx}",
            trt_prefix=f"shared_block.layers.{layer_idx}",
            dtype=dtype,
        )
    logger.info("✓ shared_block (32 layers)")
    
    # === 3. Text Block (4 层) ===
    logger.info("Converting text_block (4 layers)...")
    for layer_idx in range(4):
        _convert_decoder_layer(
            hf_weights=hf_weights,
            trtllm_weights=trtllm_weights,
            hf_prefix=f"model.text_block.layers.{layer_idx}",
            trt_prefix=f"text_block.layers.{layer_idx}",
            dtype=dtype,
        )
    logger.info("✓ text_block (4 layers)")
    
    # === 4. Audio Block (4 层) ===
    logger.info("Converting audio_block (4 layers)...")
    for layer_idx in range(4):
        _convert_decoder_layer(
            hf_weights=hf_weights,
            trtllm_weights=trtllm_weights,
            hf_prefix=f"model.audio_block.layers.{layer_idx}",
            trt_prefix=f"audio_block.layers.{layer_idx}",
            dtype=dtype,
        )
    logger.info("✓ audio_block (4 layers)")
    
    # === 5. Final Norms ===
    if "model.text_norm.weight" in hf_weights:
        trtllm_weights["text_norm.weight"] = hf_weights["model.text_norm.weight"]
        logger.info("✓ text_norm")
    
    if "model.audio_norm.weight" in hf_weights:
        trtllm_weights["audio_norm.weight"] = hf_weights["model.audio_norm.weight"]
        logger.info("✓ audio_norm")
    
    # === 6. LM Heads ===
    if "text_lm_head.weight" in hf_weights:
        trtllm_weights["text_lm_head.weight"] = hf_weights["text_lm_head.weight"]
        logger.info("✓ text_lm_head")
    
    if "audio_lm_head.weight" in hf_weights:
        trtllm_weights["audio_lm_head.weight"] = hf_weights["audio_lm_head.weight"]
        logger.info("✓ audio_lm_head")
    
    logger.info(f"Total TRT-LLM weights: {len(trtllm_weights)}")
    return trtllm_weights


def _convert_decoder_layer(
    hf_weights: Dict[str, torch.Tensor],
    trtllm_weights: Dict[str, torch.Tensor],
    hf_prefix: str,
    trt_prefix: str,
    dtype: torch.dtype,
):
    """转换单个 Decoder 层"""
    
    # Attention QKV
    q_key = f"{hf_prefix}.self_attn.q_proj.weight"
    k_key = f"{hf_prefix}.self_attn.k_proj.weight"
    v_key = f"{hf_prefix}.self_attn.v_proj.weight"
    
    if q_key in hf_weights and k_key in hf_weights and v_key in hf_weights:
        q = hf_weights[q_key]
        k = hf_weights[k_key]
        v = hf_weights[v_key]
        
        # 合并 QKV
        qkv = torch.cat([q, k, v], dim=0)
        trtllm_weights[f"{trt_prefix}.attention.qkv.weight"] = qkv
        
        # 添加零 bias
        qkv_bias = torch.zeros(qkv.shape[0], dtype=dtype)
        trtllm_weights[f"{trt_prefix}.attention.qkv.bias"] = qkv_bias
    
    # Attention Dense (O proj)
    o_key = f"{hf_prefix}.self_attn.o_proj.weight"
    if o_key in hf_weights:
        trtllm_weights[f"{trt_prefix}.attention.dense.weight"] = hf_weights[o_key]
    
    # MLP
    gate_key = f"{hf_prefix}.mlp.gate_proj.weight"
    if gate_key in hf_weights:
        trtllm_weights[f"{trt_prefix}.mlp.gate.weight"] = hf_weights[gate_key]
    
    up_key = f"{hf_prefix}.mlp.up_proj.weight"
    if up_key in hf_weights:
        trtllm_weights[f"{trt_prefix}.mlp.fc.weight"] = hf_weights[up_key]
    
    down_key = f"{hf_prefix}.mlp.down_proj.weight"
    if down_key in hf_weights:
        trtllm_weights[f"{trt_prefix}.mlp.proj.weight"] = hf_weights[down_key]
    
    # LayerNorms
    input_ln_key = f"{hf_prefix}.input_layernorm.weight"
    if input_ln_key in hf_weights:
        trtllm_weights[f"{trt_prefix}.input_layernorm.weight"] = hf_weights[input_ln_key]
    
    post_ln_key = f"{hf_prefix}.post_attention_layernorm.weight"
    if post_ln_key in hf_weights:
        trtllm_weights[f"{trt_prefix}.post_layernorm.weight"] = hf_weights[post_ln_key]


def save_full_checkpoint(
    weights: Dict[str, torch.Tensor],
    output_dir: str,
) -> str:
    """保存完整 checkpoint"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存权重
    checkpoint_path = output_path / "rank0.safetensors"
    logger.info(f"Saving checkpoint to {checkpoint_path}")
    save_file(weights, str(checkpoint_path))
    
    # 保存配置
    config = {
        "architecture": "MossSpeechForCausalLM",
        "dtype": "float16",
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_shared_layers": 32,
        "num_text_layers": 4,
        "num_audio_layers": 4,
        "vocab_size": 151680,
        "audio_vocab_size": 16512,
        "max_position_embeddings": 40960,
        "rotary_base": 10000.0,
        "mapping": {
            "tp_size": 1,
            "pp_size": 1,
        },
    }
    
    config_path = output_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"✅ Saved: {checkpoint_path}")
    return str(output_path)


def main():
    """主转换流程"""
    logger.info("=" * 60)
    logger.info("MOSS-Speech 完整权重转换 (32+4+4)")
    logger.info("=" * 60)
    
    hf_model_path = "/workspace/models/MOSS-Speech"
    output_dir = "/workspace/models/MOSS-Speech-TRTLLM-Full"
    
    # 1. 加载权重
    logger.info(f"Loading from {hf_model_path}")
    hf_weights = load_moss_weights(hf_model_path, torch.float16)
    
    # 2. 转换
    logger.info("Converting to TRT-LLM format (full architecture)...")
    trtllm_weights = convert_full_moss_speech(hf_weights, torch.float16)
    
    # 3. 保存
    output = save_full_checkpoint(trtllm_weights, output_dir)
    
    logger.info(f"✅ Conversion complete: {output}")
    
    # 4. 统计
    logger.info("\n=== 权重统计 ===")
    shared_count = len([k for k in trtllm_weights if k.startswith("shared_block")])
    text_count = len([k for k in trtllm_weights if k.startswith("text_block")])
    audio_count = len([k for k in trtllm_weights if k.startswith("audio_block")])
    other_count = len(trtllm_weights) - shared_count - text_count - audio_count
    
    logger.info(f"  shared_block: {shared_count} tensors")
    logger.info(f"  text_block: {text_count} tensors")
    logger.info(f"  audio_block: {audio_count} tensors")
    logger.info(f"  其他: {other_count} tensors")
    logger.info(f"  总计: {len(trtllm_weights)} tensors")


if __name__ == "__main__":
    main()



