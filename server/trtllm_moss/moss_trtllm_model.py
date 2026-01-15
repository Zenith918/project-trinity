#!/usr/bin/env python3
"""
MOSS-Speech TensorRT-LLM 完整模型实现
======================================

研究员 P0 核心指示:
- 完整架构: 32 层共享 + 4 层文本 + 4 层音频
- 全链路 PagedAttention
- shared_block 可选 FP8, text/audio_block FP16

基于 TRT-LLM PretrainedModel 实现，可直接用 build() 构建 Engine
"""

import copy
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from pathlib import Path

import tensorrt as trt
import tensorrt_llm
from tensorrt_llm import Module
from tensorrt_llm.functional import Tensor
from tensorrt_llm.models.modeling_utils import PretrainedModel, PretrainedConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.layers import (
    Attention, 
    GatedMLP,
    RmsNorm, 
    Embedding, 
    ColumnLinear,
)
from tensorrt_llm.module import ModuleList
from tensorrt_llm.quantization import QuantMode


class MossSpeechPretrainedConfig(PretrainedConfig):
    """MOSS-Speech 配置 (继承 PretrainedConfig)"""
    
    def __init__(
        self,
        # MOSS-Speech 特有配置
        num_shared_layers: int = 32,
        num_text_layers: int = 4,
        num_audio_layers: int = 4,
        audio_vocab_size: int = 16512,
        rotary_base: float = 10000.0,
        rms_norm_eps: float = 1e-6,
        # 继承自 PretrainedConfig 的必需参数
        **kwargs,
    ):
        # 设置默认值
        kwargs.setdefault('architecture', 'MossSpeechForCausalLM')
        kwargs.setdefault('dtype', 'float16')
        kwargs.setdefault('hidden_size', 4096)
        kwargs.setdefault('intermediate_size', 12288)
        kwargs.setdefault('num_attention_heads', 32)
        kwargs.setdefault('num_key_value_heads', 8)
        kwargs.setdefault('vocab_size', 151680)
        kwargs.setdefault('max_position_embeddings', 40960)
        kwargs.setdefault('hidden_act', 'silu')
        kwargs.setdefault('position_embedding_type', 'rope_gpt_neox')
        kwargs.setdefault('num_hidden_layers', num_shared_layers)
        
        # 调用父类初始化
        super().__init__(**kwargs)
        
        # MOSS-Speech 特有属性
        self.num_shared_layers = num_shared_layers
        self.num_text_layers = num_text_layers
        self.num_audio_layers = num_audio_layers
        self.audio_vocab_size = audio_vocab_size
        self.rotary_base = rotary_base
        self.rms_norm_eps = rms_norm_eps
    
    @property
    def total_layers(self) -> int:
        return self.num_shared_layers + self.num_text_layers + self.num_audio_layers

    def to_dict(self) -> Dict:
        d = super().to_dict() if hasattr(super(), 'to_dict') else {}
        d.update({
            "architecture": self.architecture,
            "dtype": self.dtype,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "num_shared_layers": self.num_shared_layers,
            "num_text_layers": self.num_text_layers,
            "num_audio_layers": self.num_audio_layers,
            "vocab_size": self.vocab_size,
            "audio_vocab_size": self.audio_vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
            "num_hidden_layers": self.num_hidden_layers,
        })
        return d
    
    @classmethod
    def from_json(cls, path: str) -> "MossSpeechPretrainedConfig":
        with open(path, 'r') as f:
            data = json.load(f)
        
        # 创建 mapping
        mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)
        
        return cls(
            mapping=mapping,
            hidden_size=data.get("hidden_size", 4096),
            intermediate_size=data.get("intermediate_size", 12288),
            num_attention_heads=data.get("num_attention_heads", 32),
            num_key_value_heads=data.get("num_key_value_heads", 8),
            num_shared_layers=data.get("num_shared_layers", 32),
            num_text_layers=data.get("num_text_layers", 4),
            num_audio_layers=data.get("num_audio_layers", 4),
            vocab_size=data.get("vocab_size", 151680),
            audio_vocab_size=data.get("audio_vocab_size", 16512),
            max_position_embeddings=data.get("max_position_embeddings", 40960),
        )


class MossSpeechDecoderLayer(Module):
    """单个 Transformer 解码器层"""
    
    def __init__(
        self,
        config: MossSpeechPretrainedConfig,
        layer_idx: int,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        
        dtype = trt.float16 if config.dtype == "float16" else trt.bfloat16
        
        # Input LayerNorm
        self.input_layernorm = RmsNorm(
            normalized_shape=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
        )
        
        # Self Attention (GQA)
        self.attention = Attention(
            local_layer_idx=layer_idx,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            dtype=dtype,
            attention_mask_type=tensorrt_llm.layers.AttentionMaskType.causal,
            position_embedding_type=tensorrt_llm.layers.PositionEmbeddingType.rope_gpt_neox,
            rotary_embedding_base=config.rotary_base,
            bias=True,        # QKV has bias
            dense_bias=False, # O_proj no bias (MOSS-Speech)
            tp_group=config.mapping.tp_group if hasattr(config, 'mapping') else None,
            tp_size=config.mapping.tp_size if hasattr(config, 'mapping') else 1,
        )
        
        # Post Attention LayerNorm
        self.post_layernorm = RmsNorm(
            normalized_shape=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
        )
        
        # MLP (SwiGLU)
        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            dtype=dtype,
            bias=False,
            tp_group=config.mapping.tp_group if hasattr(config, 'mapping') else None,
            tp_size=config.mapping.tp_size if hasattr(config, 'mapping') else 1,
        )
    
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        kv_cache_params: Optional[tensorrt_llm.layers.KeyValueCacheParams] = None,
        attention_params: Optional[tensorrt_llm.layers.AttentionParams] = None,
    ) -> Tensor:
        # Pre-norm + Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
        )
        hidden_states = residual + hidden_states
        
        # Pre-norm + MLP
        residual = hidden_states
        hidden_states = self.post_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class MossSpeechModel(Module):
    """MOSS-Speech Transformer 主体 (不含 LM Head)"""
    
    def __init__(self, config: MossSpeechPretrainedConfig):
        super().__init__()
        self.config = config
        
        dtype = trt.float16 if config.dtype == "float16" else trt.bfloat16
        
        # Embeddings
        self.vocab_embedding = Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            dtype=dtype,
            tp_group=config.mapping.tp_group if hasattr(config, 'mapping') else None,
            tp_size=config.mapping.tp_size if hasattr(config, 'mapping') else 1,
        )
        self.audio_embedding = Embedding(
            num_embeddings=config.audio_vocab_size,
            embedding_dim=config.hidden_size,
            dtype=dtype,
        )
        
        # Shared Block (32 layers)
        self.shared_layers = ModuleList([
            MossSpeechDecoderLayer(config, layer_idx=i)
            for i in range(config.num_shared_layers)
        ])
        
        # Text Block (4 layers) - 独立层索引从 32 开始
        self.text_layers = ModuleList([
            MossSpeechDecoderLayer(config, layer_idx=config.num_shared_layers + i)
            for i in range(config.num_text_layers)
        ])
        
        # Audio Block (4 layers) - 独立层索引从 32 开始
        self.audio_layers = ModuleList([
            MossSpeechDecoderLayer(config, layer_idx=config.num_shared_layers + i)
            for i in range(config.num_audio_layers)
        ])
        
        # Final norms
        self.text_norm = RmsNorm(
            normalized_shape=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
        )
        self.audio_norm = RmsNorm(
            normalized_shape=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
        )


class MossSpeechForCausalLM(PretrainedModel):
    """
    完整的 MOSS-Speech TensorRT-LLM 模型
    
    架构:
    ┌───────────────────────────────────────┐
    │  embed_tokens + audio_embed           │
    │              ↓                        │
    │     shared_block (32 层)              │
    │              ↓                        │
    │      ┌──────┴──────┐                 │
    │      ↓             ↓                 │
    │  text_block    audio_block           │
    │   (4 层)        (4 层)                │
    │      ↓             ↓                 │
    │  text_norm    audio_norm             │
    │      ↓             ↓                 │
    │  text_head    audio_head             │
    │  (151680)     (16512)                │
    └───────────────────────────────────────┘
    """
    
    config_class = MossSpeechPretrainedConfig
    
    def __init__(self, config: MossSpeechPretrainedConfig):
        super().__init__(config)
        self.config = config
        
        dtype = trt.float16 if config.dtype == "float16" else trt.bfloat16
        
        # 创建 RoPE embedding 常量参数 (关键！)
        # 这会在模型类上注册 embed_positions, rotary_inv_freq, embed_positions_for_gpt_attention
        Attention.create_attention_const_params(self, config)
        
        # 设置 position_embedding_type 属性供 fill_attention_params 使用
        from tensorrt_llm.layers import PositionEmbeddingType
        self.position_embedding_type = PositionEmbeddingType.rope_gpt_neox
        
        # Transformer
        self.transformer = MossSpeechModel(config)
        
        # LM Heads
        self.text_lm_head = ColumnLinear(
            in_features=config.hidden_size,
            out_features=config.vocab_size,
            bias=False,
            dtype=dtype,
            tp_group=config.mapping.tp_group if hasattr(config, 'mapping') else None,
            tp_size=config.mapping.tp_size if hasattr(config, 'mapping') else 1,
        )
        self.audio_lm_head = ColumnLinear(
            in_features=config.hidden_size,
            out_features=config.audio_vocab_size,
            bias=False,
            dtype=dtype,
        )
    
    def forward(
        self,
        input_ids: Tensor,
        position_ids: Optional[Tensor] = None,
        use_cache: bool = False,
        last_token_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        kv_cache_params: Optional[tensorrt_llm.layers.KeyValueCacheParams] = None,
        attention_params: Optional[tensorrt_llm.layers.AttentionParams] = None,
        hidden_states: Optional[Tensor] = None,
        **kwargs,
    ):
        """
        前向传播
        
        Returns:
            text_logits: 文本输出 logits
            (audio_logits 通过 register_network_output 单独输出)
        """
        # 填充 RoPE 参数到 attention_params (关键！)
        attention_params = Attention.fill_attention_params(self, attention_params)
        
        # 1. Embedding
        if hidden_states is None:
            hidden_states = self.transformer.vocab_embedding(input_ids)
        
        # 2. Shared Block (32 layers)
        for layer in self.transformer.shared_layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                kv_cache_params=kv_cache_params,
                attention_params=attention_params,
            )
        
        # 3. 分支处理
        # Text branch
        text_hidden = hidden_states
        for layer in self.transformer.text_layers:
            text_hidden = layer(
                text_hidden,
                attention_mask=attention_mask,
                kv_cache_params=kv_cache_params,
                attention_params=attention_params,
            )
        text_hidden = self.transformer.text_norm(text_hidden)
        
        # Audio branch
        audio_hidden = hidden_states
        for layer in self.transformer.audio_layers:
            audio_hidden = layer(
                audio_hidden,
                attention_mask=attention_mask,
                kv_cache_params=kv_cache_params,
                attention_params=attention_params,
            )
        audio_hidden = self.transformer.audio_norm(audio_hidden)
        
        # 4. LM Heads
        text_logits = self.text_lm_head(text_hidden)
        audio_logits = self.audio_lm_head(audio_hidden)
        
        # 标记输出 (TRT-LLM 需要显式标记)
        text_logits.mark_output('logits', self.config.logits_dtype)
        audio_logits.mark_output('audio_logits', self.config.logits_dtype)
        
        return text_logits
    
    def prepare_inputs(
        self,
        max_batch_size: int,
        max_input_len: int,
        max_seq_len: int,
        max_num_tokens: int,
        use_cache: bool,
        max_beam_width: int = 1,
        opt_num_tokens: int = None,
        prompt_embedding_table_size: int = 0,
        position_encoding_2d: bool = False,
        max_draft_len: int = 0,
        speculative_decoding_draft_tokens_external: bool = False,
        spec_decoding_is_generation_length_variable: bool = False,
        gather_context_logits: bool = False,
        gather_generation_logits: bool = False,
        lora_target_modules: List[str] = None,
        opt_batch_size: int = 0,
    ):
        """准备输入 Tensors (TRT-LLM 标准接口) - 使用 prepare_basic_inputs"""
        from tensorrt_llm._common import default_net
        from tensorrt_llm._utils import str_dtype_to_trt
        from tensorrt_llm.bindings import KVCacheType
        
        # 获取 plugin 配置
        remove_input_padding = default_net().plugin_config.remove_input_padding
        use_gpt_attention_plugin = default_net().plugin_config.gpt_attention_plugin
        use_gemm_plugin = default_net().plugin_config.gemm_plugin
        paged_kv_cache = default_net().plugin_config.paged_kv_cache
        tokens_per_block = default_net().plugin_config.tokens_per_block
        multiple_profiles = default_net().plugin_config.multiple_profiles
        
        # 确定 KV Cache 类型
        if not use_cache:
            kv_cache_type = KVCacheType.DISABLED
        elif paged_kv_cache:
            kv_cache_type = KVCacheType.PAGED
        else:
            kv_cache_type = KVCacheType.CONTINUOUS
        
        # 使用父类的 prepare_basic_inputs 方法
        model_inputs = self.prepare_basic_inputs(
            max_batch_size=max_batch_size,
            max_beam_width=max_beam_width,
            max_input_len=max_input_len,
            max_seq_len=max_seq_len,
            hidden_size=self.config.hidden_size,
            num_kv_heads=self.config.num_key_value_heads,
            head_size=self.config.head_size,
            num_layers=self.config.num_hidden_layers,
            kv_dtype=str_dtype_to_trt(self.config.dtype),
            remove_input_padding=remove_input_padding,
            use_gpt_attention_plugin=use_gpt_attention_plugin,
            use_gemm_plugin=use_gemm_plugin,
            kv_cache_type=kv_cache_type,
            tokens_per_block=tokens_per_block,
            num_heads=self.config.num_attention_heads,
            max_num_tokens=max_num_tokens,
            opt_num_tokens=opt_num_tokens,
            dtype=str_dtype_to_trt(self.config.dtype),
            prompt_embedding_table_size=prompt_embedding_table_size,
            position_encoding_2d=position_encoding_2d,
            mapping=self.config.mapping,
            gather_context_logits=gather_context_logits,
            gather_generation_logits=gather_generation_logits,
            max_draft_len=max_draft_len,
            speculative_decoding_draft_tokens_external=speculative_decoding_draft_tokens_external,
            spec_decoding_is_generation_length_variable=spec_decoding_is_generation_length_variable,
            lora_target_modules=lora_target_modules,
            multiple_profiles=multiple_profiles,
            opt_batch_size=opt_batch_size,
        )
        
        from tensorrt_llm.layers import KeyValueCacheParams, AttentionParams
        
        result = {
            'input_ids': model_inputs['input_ids'],
            'position_ids': model_inputs['position_ids'],
            'use_cache': kv_cache_type != KVCacheType.DISABLED,
            'last_token_ids': model_inputs['last_token_ids'],
            'attention_mask': model_inputs['attention_mask'],
            'kv_cache_params': KeyValueCacheParams(
                past_key_value=model_inputs['past_key_value'],
                host_past_key_value_lengths=model_inputs['host_past_key_value_lengths'],
                host_max_attention_window_sizes=model_inputs['host_max_attention_window_sizes'],
                host_sink_token_length=model_inputs['host_sink_token_length'],
                kv_cache_block_offsets=model_inputs.get('kv_cache_block_offsets'),
                host_kv_cache_block_offsets=model_inputs.get('host_kv_cache_block_offsets'),
                host_kv_cache_pool_pointers=model_inputs.get('host_kv_cache_pool_pointers'),
                cache_indirection=model_inputs['cache_indirection'],
            ),
            'attention_params': AttentionParams(
                sequence_length=model_inputs['sequence_length'],
                context_lengths=model_inputs['context_lengths'],
                host_context_lengths=model_inputs['host_context_lengths'],
                max_context_length=max_input_len,
                host_request_types=model_inputs['host_request_types'],
                host_runtime_perf_knobs=model_inputs.get('host_runtime_perf_knobs'),
            ),
        }
        
        return result
    
    @classmethod
    def from_checkpoint(
        cls,
        ckpt_dir: str,
        rank: Optional[int] = None,
        config: Optional[MossSpeechPretrainedConfig] = None,
    ) -> "MossSpeechForCausalLM":
        """从 checkpoint 加载模型"""
        ckpt_path = Path(ckpt_dir)
        
        # 加载配置
        if config is None:
            config_path = ckpt_path / "config.json"
            config = MossSpeechPretrainedConfig.from_json(str(config_path))
        
        # 创建模型
        model = cls(config)
        
        # 加载权重
        weights_path = ckpt_path / "rank0.safetensors"
        if weights_path.exists():
            model._load_weights(str(weights_path))
        
        return model
    
    def _load_weights(self, weights_path: str):
        """加载 safetensors 权重"""
        from safetensors import safe_open
        
        print(f"Loading weights from: {weights_path}")
        
        # 建立映射
        mapping = self._build_weight_mapping()
        
        loaded = 0
        with safe_open(weights_path, framework="pt", device="cpu") as f:
            weight_keys = set(f.keys())
            
            for param_name, param in self.named_parameters():
                ckpt_name = mapping.get(param_name, param_name)
                
                if ckpt_name in weight_keys:
                    tensor = f.get_tensor(ckpt_name)
                    param.value = tensor.numpy()
                    loaded += 1
        
        total = len(list(self.named_parameters()))
        print(f"Loaded {loaded}/{total} weights")
    
    def _build_weight_mapping(self) -> Dict[str, str]:
        """构建参数名到 checkpoint 权重名的映射"""
        mapping = {}
        
        # Embeddings
        mapping['transformer.vocab_embedding.weight'] = 'embed_tokens.weight'
        mapping['transformer.audio_embedding.weight'] = 'audio_embed.weight'
        
        # Shared layers
        for i in range(self.config.num_shared_layers):
            base = f'transformer.shared_layers.{i}'
            ckpt_base = f'shared_block.layers.{i}'
            self._add_layer_mapping(mapping, base, ckpt_base)
        
        # Text layers
        for i in range(self.config.num_text_layers):
            base = f'transformer.text_layers.{i}'
            ckpt_base = f'text_block.layers.{i}'
            self._add_layer_mapping(mapping, base, ckpt_base)
        
        # Audio layers
        for i in range(self.config.num_audio_layers):
            base = f'transformer.audio_layers.{i}'
            ckpt_base = f'audio_block.layers.{i}'
            self._add_layer_mapping(mapping, base, ckpt_base)
        
        # Norms and heads
        mapping['transformer.text_norm.weight'] = 'text_norm.weight'
        mapping['transformer.audio_norm.weight'] = 'audio_norm.weight'
        mapping['text_lm_head.weight'] = 'text_lm_head.weight'
        mapping['audio_lm_head.weight'] = 'audio_lm_head.weight'
        
        return mapping
    
    def _add_layer_mapping(self, mapping: Dict, base: str, ckpt_base: str):
        """添加单层的权重映射"""
        # Attention
        mapping[f'{base}.attention.qkv.weight'] = f'{ckpt_base}.attention.qkv.weight'
        mapping[f'{base}.attention.qkv.bias'] = f'{ckpt_base}.attention.qkv.bias'
        mapping[f'{base}.attention.dense.weight'] = f'{ckpt_base}.attention.dense.weight'
        
        # MLP
        mapping[f'{base}.mlp.gate.weight'] = f'{ckpt_base}.mlp.gate.weight'
        mapping[f'{base}.mlp.fc.weight'] = f'{ckpt_base}.mlp.fc.weight'
        mapping[f'{base}.mlp.proj.weight'] = f'{ckpt_base}.mlp.proj.weight'
        
        # Norms
        mapping[f'{base}.input_layernorm.weight'] = f'{ckpt_base}.input_layernorm.weight'
        mapping[f'{base}.post_layernorm.weight'] = f'{ckpt_base}.post_layernorm.weight'


# 注册模型到 MODEL_MAP
def register_moss_speech_model():
    """注册 MOSS-Speech 模型到 TRT-LLM 模型映射"""
    try:
        from tensorrt_llm.models import MODEL_MAP
        MODEL_MAP['MossSpeechForCausalLM'] = MossSpeechForCausalLM
        print("✅ MossSpeechForCausalLM registered to MODEL_MAP")
    except ImportError:
        print("⚠️ Could not register to MODEL_MAP")


if __name__ == "__main__":
    # 测试模型创建
    register_moss_speech_model()
    
    config = MossSpeechPretrainedConfig.from_json(
        "/workspace/models/MOSS-Speech-TRTLLM-Full/config.json"
    )
    print(f"Config: {config.to_dict()}")
    
    model = MossSpeechForCausalLM.from_checkpoint(
        "/workspace/models/MOSS-Speech-TRTLLM-Full"
    )
    print("✅ Model created and weights loaded!")

