"""
MOSS-Speech 完整 TensorRT-LLM 模型
===================================

研究员 P0 指示: 使用 Python API 构建完整架构 (32+4+4)

架构:
- shared_block: 32 层共享 Transformer (FP8 量化候选)
- text_block: 4 层文本专用 (FP16)
- audio_block: 4 层音频专用 (FP16)
- text_head: 文本输出头
- audio_head: 音频输出头
"""

import math
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import tensorrt as trt
import tensorrt_llm
from tensorrt_llm import Module, Tensor
from tensorrt_llm.layers import (
    Attention, 
    MLP, 
    GatedMLP,
    RmsNorm, 
    Embedding, 
    Linear,
    ColumnLinear,
    RowLinear,
)
from tensorrt_llm.parameter import Parameter
from tensorrt_llm.functional import (
    concat, 
    split, 
    shape, 
    gather,
    slice,
    unsqueeze,
    expand,
    matmul,
    softmax,
    gelu,
    silu,
    cast,
)
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.mapping import Mapping
import tensorrt_llm.functional as F


@dataclass
class MossSpeechConfig:
    """MOSS-Speech 配置"""
    # 基础配置
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    
    # 层数配置
    num_shared_layers: int = 32
    num_text_layers: int = 4
    num_audio_layers: int = 4
    
    # 词表配置
    vocab_size: int = 151680
    audio_vocab_size: int = 16512
    
    # 位置编码
    max_position_embeddings: int = 40960
    rotary_base: float = 10000.0
    
    # 数据类型
    dtype: str = "float16"  # shared_block 可设为 fp8
    
    # 量化
    use_fp8_shared: bool = False  # 研究员 P1 指示
    
    @property
    def total_layers(self) -> int:
        return self.num_shared_layers + self.num_text_layers + self.num_audio_layers


class MossSpeechDecoderLayer(Module):
    """
    单个 Transformer 解码器层
    
    与 Qwen2 结构一致:
    - RMSNorm + Attention + RMSNorm + MLP
    - GQA (Grouped Query Attention)
    - SwiGLU MLP
    """
    
    def __init__(
        self,
        config: MossSpeechConfig,
        layer_idx: int,
        dtype: trt.DataType = trt.float16,
        quant_mode: QuantMode = QuantMode(0),
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.dtype = dtype
        
        # Input LayerNorm
        self.input_layernorm = RmsNorm(
            normalized_shape=config.hidden_size,
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
            quant_mode=quant_mode,
        )
        
        # Post Attention LayerNorm
        self.post_attention_layernorm = RmsNorm(
            normalized_shape=config.hidden_size,
            dtype=dtype,
        )
        
        # MLP (SwiGLU)
        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.intermediate_size,
            hidden_act='silu',
            dtype=dtype,
            quant_mode=quant_mode,
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
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class MossSpeechTransformerBlock(Module):
    """
    Transformer 块 (多层)
    
    用于:
    - shared_block (32 层)
    - text_block (4 层)
    - audio_block (4 层)
    """
    
    def __init__(
        self,
        config: MossSpeechConfig,
        num_layers: int,
        start_layer_idx: int = 0,
        dtype: trt.DataType = trt.float16,
        quant_mode: QuantMode = QuantMode(0),
        block_name: str = "shared",
    ):
        super().__init__()
        self.config = config
        self.num_layers = num_layers
        self.block_name = block_name
        
        # 创建多层
        self.layers = tensorrt_llm.module.ModuleList([
            MossSpeechDecoderLayer(
                config=config,
                layer_idx=start_layer_idx + i,
                dtype=dtype,
                quant_mode=quant_mode,
            )
            for i in range(num_layers)
        ])
    
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        kv_cache_params: Optional[tensorrt_llm.layers.KeyValueCacheParams] = None,
        attention_params: Optional[tensorrt_llm.layers.AttentionParams] = None,
    ) -> Tensor:
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                kv_cache_params=kv_cache_params,
                attention_params=attention_params,
            )
        return hidden_states


class MossSpeechForCausalLM(Module):
    """
    完整的 MOSS-Speech 模型
    
    架构:
    ┌─────────────────────────────────────────┐
    │  embed_tokens + audio_embed             │
    │              ↓                          │
    │     shared_block (32 层)                │
    │              ↓                          │
    │      ┌──────┴──────┐                   │
    │      ↓             ↓                   │
    │  text_block    audio_block             │
    │   (4 层)        (4 层)                  │
    │      ↓             ↓                   │
    │  text_norm    audio_norm               │
    │      ↓             ↓                   │
    │  text_head    audio_head               │
    └─────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        config: MossSpeechConfig,
        mapping: Mapping = Mapping(),
    ):
        super().__init__()
        self.config = config
        self.mapping = mapping
        
        # 数据类型
        dtype = trt.float16 if config.dtype == "float16" else trt.bfloat16
        
        # 量化模式 (研究员 P1 指示: shared_block FP8)
        shared_quant_mode = QuantMode.from_description(
            use_fp8_qdq=config.use_fp8_shared
        ) if config.use_fp8_shared else QuantMode(0)
        
        # 音频层保持 FP16
        audio_quant_mode = QuantMode(0)
        
        # === Embeddings ===
        self.embed_tokens = Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            dtype=dtype,
        )
        self.audio_embed = Embedding(
            num_embeddings=config.audio_vocab_size,
            embedding_dim=config.hidden_size,
            dtype=dtype,
        )
        
        # === Shared Block (32 层) - 可 FP8 量化 ===
        self.shared_block = MossSpeechTransformerBlock(
            config=config,
            num_layers=config.num_shared_layers,
            start_layer_idx=0,
            dtype=dtype,
            quant_mode=shared_quant_mode,
            block_name="shared",
        )
        
        # === Text Block (4 层) - FP16 ===
        self.text_block = MossSpeechTransformerBlock(
            config=config,
            num_layers=config.num_text_layers,
            start_layer_idx=config.num_shared_layers,
            dtype=dtype,
            quant_mode=audio_quant_mode,  # FP16
            block_name="text",
        )
        
        # === Audio Block (4 层) - FP16 ===
        self.audio_block = MossSpeechTransformerBlock(
            config=config,
            num_layers=config.num_audio_layers,
            start_layer_idx=config.num_shared_layers,
            dtype=dtype,
            quant_mode=audio_quant_mode,  # FP16
            block_name="audio",
        )
        
        # === Final Norms ===
        self.text_norm = RmsNorm(
            normalized_shape=config.hidden_size,
            dtype=dtype,
        )
        self.audio_norm = RmsNorm(
            normalized_shape=config.hidden_size,
            dtype=dtype,
        )
        
        # === LM Heads ===
        self.text_lm_head = ColumnLinear(
            in_features=config.hidden_size,
            out_features=config.vocab_size,
            bias=False,
            dtype=dtype,
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
        audio_input_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        kv_cache_params: Optional[tensorrt_llm.layers.KeyValueCacheParams] = None,
        attention_params: Optional[tensorrt_llm.layers.AttentionParams] = None,
        output_text: bool = True,
        output_audio: bool = True,
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """
        前向传播
        
        Args:
            input_ids: 文本 token IDs [batch, seq_len]
            audio_input_ids: 音频 token IDs (可选)
            ...
            
        Returns:
            (text_logits, audio_logits)
        """
        # === 1. Embedding ===
        hidden_states = self.embed_tokens(input_ids)
        
        # 如果有音频输入，合并嵌入
        if audio_input_ids is not None:
            audio_hidden = self.audio_embed(audio_input_ids)
            # 这里需要根据实际需求处理合并逻辑
            # 简化版: 假设音频和文本交替
            hidden_states = hidden_states + audio_hidden
        
        # === 2. Shared Block (32 层) ===
        hidden_states = self.shared_block(
            hidden_states,
            attention_mask=attention_mask,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
        )
        
        # === 3. 分支处理 ===
        text_logits = None
        audio_logits = None
        
        if output_text:
            # Text Branch
            text_hidden = self.text_block(
                hidden_states,
                attention_mask=attention_mask,
                kv_cache_params=kv_cache_params,
                attention_params=attention_params,
            )
            text_hidden = self.text_norm(text_hidden)
            text_logits = self.text_lm_head(text_hidden)
        
        if output_audio:
            # Audio Branch
            audio_hidden = self.audio_block(
                hidden_states,
                attention_mask=attention_mask,
                kv_cache_params=kv_cache_params,
                attention_params=attention_params,
            )
            audio_hidden = self.audio_norm(audio_hidden)
            audio_logits = self.audio_lm_head(audio_hidden)
        
        return text_logits, audio_logits
    
    def prepare_inputs(
        self,
        max_batch_size: int,
        max_input_len: int,
        max_seq_len: int,
        use_cache: bool = True,
        max_beam_width: int = 1,
    ) -> Dict[str, Tensor]:
        """准备输入张量"""
        from tensorrt_llm.functional import Tensor
        
        # 输入 IDs
        input_ids = Tensor(
            name='input_ids',
            dtype=trt.int32,
            shape=[-1, -1],  # [batch, seq_len]
        )
        
        # 位置 IDs
        position_ids = Tensor(
            name='position_ids',
            dtype=trt.int32,
            shape=[-1, -1],
        )
        
        # Attention mask
        attention_mask = Tensor(
            name='attention_mask',
            dtype=trt.int32,
            shape=[-1],
        )
        
        return {
            'input_ids': input_ids,
            'position_ids': position_ids,
            'attention_mask': attention_mask,
        }


def build_moss_speech_engine(
    config: MossSpeechConfig,
    output_dir: str,
    max_batch_size: int = 1,
    max_input_len: int = 2048,
    max_seq_len: int = 6144,
) -> str:
    """
    使用 Python API 构建完整 MOSS-Speech Engine
    
    研究员指示: 弃用 trtllm-build，改用 Python API
    """
    from tensorrt_llm.builder import Builder
    from tensorrt_llm.network import net_guard
    from tensorrt_llm.plugin import PluginConfig
    import os
    
    print("=" * 60)
    print("Building MOSS-Speech TensorRT Engine (Full Architecture)")
    print("=" * 60)
    print(f"  - Shared layers: {config.num_shared_layers}")
    print(f"  - Text layers: {config.num_text_layers}")
    print(f"  - Audio layers: {config.num_audio_layers}")
    print(f"  - FP8 for shared: {config.use_fp8_shared}")
    
    # 创建模型
    mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)
    model = MossSpeechForCausalLM(config, mapping)
    
    # 创建 Builder
    builder = Builder()
    
    # Plugin 配置
    plugin_config = PluginConfig()
    plugin_config.gpt_attention_plugin = 'float16'
    plugin_config.gemm_plugin = 'float16'
    plugin_config.paged_kv_cache = True
    plugin_config.remove_input_padding = True
    plugin_config.context_fmha = True
    
    # 构建配置
    builder_config = builder.create_builder_config(
        name='moss_speech',
        precision='float16',
        timing_cache=None,
        profiling_verbosity='detailed',
    )
    
    # 构建网络
    with net_guard(builder.create_network()) as network:
        # 准备输入
        inputs = model.prepare_inputs(
            max_batch_size=max_batch_size,
            max_input_len=max_input_len,
            max_seq_len=max_seq_len,
        )
        
        # 前向传播
        text_logits, audio_logits = model(**inputs)
        
        # 标记输出
        if text_logits is not None:
            text_logits.mark_output('text_logits', trt.float32)
        if audio_logits is not None:
            audio_logits.mark_output('audio_logits', trt.float32)
        
        # 构建 Engine
        engine = builder.build_engine(network, builder_config)
    
    # 保存 Engine
    os.makedirs(output_dir, exist_ok=True)
    engine_path = os.path.join(output_dir, 'moss_speech_full.engine')
    
    with open(engine_path, 'wb') as f:
        f.write(engine)
    
    print(f"\n✅ Engine saved to: {engine_path}")
    return engine_path


if __name__ == "__main__":
    # 测试模型构建
    config = MossSpeechConfig(
        num_shared_layers=32,
        num_text_layers=4,
        num_audio_layers=4,
        use_fp8_shared=False,  # 先测试 FP16
    )
    
    print("MOSS-Speech Config:")
    print(f"  - Total layers: {config.total_layers}")
    print(f"  - Hidden size: {config.hidden_size}")
    print(f"  - Vocab size: {config.vocab_size}")
    print(f"  - Audio vocab size: {config.audio_vocab_size}")



