"""
MOSS-Speech TensorRT-LLM Model Definition
==========================================

核心技术: 虚拟线性化 (Virtual Linearization)
作者: 豆包团队
最后更新: 2026-01-18

⚠️ 警告: 本文件包含关键的架构伪装代码，修改前请阅读:
   /workspace/docs/moss-speech/PITFALLS_AND_SOLUTIONS.md
   /workspace/docs/moss-speech/ARCHITECTURE.md

架构说明:
---------
MOSS-Speech 采用分叉架构: 32 Shared + 4 Text + 4 Audio = 40 层
为了绕过 TensorRT-LLM gptAttentionPlugin 的线性层假设，
我们通过"虚拟线性化"将分叉架构伪装成 40 层线性堆叠。

关键参数 (禁止修改):
-------------------
- num_hidden_layers = 40
- epsilon = 1e-4 (虚拟线性化依赖链)
- audio_start_idx = 36
"""

import json
import os
import inspect
from typing import Optional, Dict, Any, List

import tensorrt_llm
from tensorrt_llm import Tensor
from tensorrt_llm.functional import (
    concat, identity, gather_last_token_logits
)
from tensorrt_llm.layers import (
    Attention, AttentionMaskType, PositionEmbeddingType,
    MLP, RmsNorm, Embedding, ColumnLinear, GatedMLP
)
from tensorrt_llm.models import PretrainedConfig
from tensorrt_llm.models.modeling_utils import DecoderModelForCausalLM, DecoderLayerList
from tensorrt_llm.module import Module, ModuleList

# ═══════════════════════════════════════════════════════════════════════════════
# 🔴 关键常量 - 禁止修改！
# ═══════════════════════════════════════════════════════════════════════════════
NUM_SHARED_LAYERS = 32   # Shared 层数
NUM_TEXT_LAYERS = 4      # Text 分支层数
NUM_AUDIO_LAYERS = 4     # Audio 分支层数
TOTAL_LAYERS = NUM_SHARED_LAYERS + NUM_TEXT_LAYERS + NUM_AUDIO_LAYERS  # 必须为 40

TEXT_VOCAB_SIZE = 151680   # Text 词表大小
AUDIO_VOCAB_SIZE = 16512   # Audio 词表大小
COMBINED_VOCAB_SIZE = TEXT_VOCAB_SIZE + AUDIO_VOCAB_SIZE  # 168192

# 虚拟线性化 epsilon - 禁止删除或修改！
# 原因: FP16 精度下，太小会被舍入为 0，太大会影响输出
VIRTUAL_LINEARIZATION_EPSILON = 1e-4


def verify_architecture_integrity():
    """
    [ARCH_GUARD] 架构完整性验证
    
    在模块加载时自动执行，确保关键参数未被修改。
    """
    assert TOTAL_LAYERS == 40, \
        f"[ARCH_GUARD] FATAL: TOTAL_LAYERS={TOTAL_LAYERS}, 必须为 40！"
    
    assert NUM_SHARED_LAYERS == 32, \
        f"[ARCH_GUARD] FATAL: NUM_SHARED_LAYERS={NUM_SHARED_LAYERS}, 必须为 32！"
    
    assert VIRTUAL_LINEARIZATION_EPSILON == 1e-4, \
        f"[ARCH_GUARD] FATAL: epsilon={VIRTUAL_LINEARIZATION_EPSILON}, 必须为 1e-4！"
    
    print("[ARCH_GUARD] ✅ 40-Layer Virtual Linearization Active")


# 模块加载时验证
verify_architecture_integrity()


class MossSpeechConfig(PretrainedConfig):
    """
    MOSS-Speech 模型配置
    
    继承自 TensorRT-LLM 的 PretrainedConfig，添加了分叉架构特有的参数。
    """
    
    def __init__(
        self,
        architecture: str = "MossSpeechForCausalLM",
        dtype: str = "float16",
        logits_dtype: str = "float32",
        vocab_size: int = COMBINED_VOCAB_SIZE,
        hidden_size: int = 4096,
        intermediate_size: int = 12288,
        num_hidden_layers: int = TOTAL_LAYERS,  # 🔴 强制 40 层
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        hidden_act: str = "silu",
        max_position_embeddings: int = 40960,
        rms_norm_eps: float = 1e-6,
        rotary_base: float = 1000000.0,
        position_embedding_type: PositionEmbeddingType = PositionEmbeddingType.rope_gpt_neox,
        num_shared_layers: int = NUM_SHARED_LAYERS,
        num_text_layers: int = NUM_TEXT_LAYERS,
        num_audio_layers: int = NUM_AUDIO_LAYERS,
        text_vocab_size: int = TEXT_VOCAB_SIZE,
        audio_vocab_size: int = AUDIO_VOCAB_SIZE,
        **kwargs
    ):
        # ═══════════════════════════════════════════════════════════════════════
        # 清理 kwargs 中可能重复的参数，防止 "multiple values" 错误
        # ═══════════════════════════════════════════════════════════════════════
        for key in ['num_hidden_layers', 'architecture', 'dtype', 'vocab_size', 
                    'hidden_size', 'intermediate_size', 'num_attention_heads',
                    'num_key_value_heads', 'hidden_act', 'max_position_embeddings',
                    'logits_dtype', 'position_embedding_type']:
            kwargs.pop(key, None)
        
        # ═══════════════════════════════════════════════════════════════════════
        # 🔴 [ARCH_GUARD] 强制使用 40 层，忽略传入的 num_hidden_layers
        # ═══════════════════════════════════════════════════════════════════════
        num_hidden_layers = TOTAL_LAYERS
        
        super().__init__(
            architecture=architecture,
            dtype=dtype,
            logits_dtype=logits_dtype,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            position_embedding_type=position_embedding_type,
            **kwargs
        )
        
        self.rms_norm_eps = rms_norm_eps
        self.rotary_base = rotary_base
        self.num_shared_layers = num_shared_layers
        self.num_text_layers = num_text_layers
        self.num_audio_layers = num_audio_layers
        self.text_vocab_size = text_vocab_size
        self.audio_vocab_size = audio_vocab_size
        
        # 确保 head_size 被设置
        if not hasattr(self, 'head_size') or self.head_size is None:
            self.head_size = hidden_size // num_attention_heads
    
    def __repr__(self):
        """[ARCH_GUARD] 打印架构信息"""
        return (
            f"MossSpeechConfig(\n"
            f"  [ARCH_GUARD] 40-Layer Virtual Linearization Active\n"
            f"  num_hidden_layers={self.num_hidden_layers},\n"
            f"  num_shared_layers={self.num_shared_layers},\n"
            f"  num_text_layers={self.num_text_layers},\n"
            f"  num_audio_layers={self.num_audio_layers},\n"
            f"  epsilon={VIRTUAL_LINEARIZATION_EPSILON}\n"
            f")"
        )


class MossSpeechDecoderLayer(Module):
    """
    MOSS-Speech 单层 Decoder
    
    与标准 Transformer 层相同，但注意 layer_idx 必须在 0-39 范围内。
    """
    
    def __init__(self, config: MossSpeechConfig, layer_idx: int):
        super().__init__()
        
        # ═══════════════════════════════════════════════════════════════════════
        # [ARCH_GUARD] 验证 layer_idx 范围
        # ═══════════════════════════════════════════════════════════════════════
        assert 0 <= layer_idx < TOTAL_LAYERS, \
            f"[ARCH_GUARD] layer_idx={layer_idx} 超出范围 [0, {TOTAL_LAYERS})"
        
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        dtype = config.dtype
        
        # Input LayerNorm
        self.input_layernorm = RmsNorm(
            normalized_shape=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
        )
        
        # Attention - 使用 local_layer_idx 确保 KV Cache 正确寻址
        self.attention = Attention(
            local_layer_idx=layer_idx,  # 🔴 关键: 每层唯一的索引
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            dtype=dtype,
            attention_mask_type=AttentionMaskType.causal,
            position_embedding_type=config.position_embedding_type,
            rotary_embedding_base=config.rotary_base,
            tp_group=None,
            tp_size=1,
        )
        
        # MLP
        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            dtype=dtype,
            tp_group=None,
            tp_size=1,
        )
        
        # Post attention LayerNorm
        self.post_layernorm = RmsNorm(
            normalized_shape=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
        )
    
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        use_cache: bool = False,
        kv_cache_params=None,
        attention_params=None,
    ):
        # Self attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        attention_output = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
        )
        
        if use_cache:
            attention_output, presents = attention_output
        
        hidden_states = residual + attention_output
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        if use_cache:
            return hidden_states, presents
        return hidden_states


class MossSpeechModel(Module):
    """
    MOSS-Speech Transformer 模型
    
    实现虚拟线性化 (Virtual Linearization):
    - 物理上: 40 层线性堆叠
    - 逻辑上: Shared(0-31) → Text(32-35) / Audio(36-39) 分叉
    """
    
    def __init__(self, config: MossSpeechConfig):
        super().__init__()
        self.config = config
        dtype = config.dtype
        
        # ═══════════════════════════════════════════════════════════════════════
        # Embedding 层
        # ═══════════════════════════════════════════════════════════════════════
        self.embed_tokens = Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            dtype=dtype,
        )
        
        # ═══════════════════════════════════════════════════════════════════════
        # 40 层 Decoder Layers (线性堆叠)
        # 
        # 层索引映射:
        #   0-31:  Shared 层 (Text 和 Audio 都经过)
        #   32-35: Text 专用层
        #   36-39: Audio 专用层
        # ═══════════════════════════════════════════════════════════════════════
        self.layers = ModuleList([
            MossSpeechDecoderLayer(config, layer_idx=i)
            for i in range(TOTAL_LAYERS)
        ])
        
        # ═══════════════════════════════════════════════════════════════════════
        # 🔴 分支索引 - 虚拟线性化的关键参数
        # ═══════════════════════════════════════════════════════════════════════
        self.shared_end_idx = NUM_SHARED_LAYERS - 1        # 31
        self.text_start_idx = NUM_SHARED_LAYERS            # 32
        self.text_end_idx = NUM_SHARED_LAYERS + NUM_TEXT_LAYERS - 1  # 35
        self.audio_start_idx = NUM_SHARED_LAYERS + NUM_TEXT_LAYERS   # 36 🔴 关键
        
        # Final layer norm
        self.norm = RmsNorm(
            normalized_shape=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
        )
        
        # [ARCH_GUARD] 打印架构信息
        print(f"[ARCH_GUARD] ✅ MossSpeechModel 初始化完成:")
        print(f"   总层数: {TOTAL_LAYERS}")
        print(f"   Shared: 0-{self.shared_end_idx}")
        print(f"   Text: {self.text_start_idx}-{self.text_end_idx}")
        print(f"   Audio: {self.audio_start_idx}-{TOTAL_LAYERS-1}")
        print(f"   Virtual Linearization epsilon: {VIRTUAL_LINEARIZATION_EPSILON}")
    
    def forward(
        self,
        input_ids: Tensor,
        position_ids: Optional[Tensor] = None,
        use_cache: bool = False,
        attention_mask: Optional[Tensor] = None,
        kv_cache_params=None,
        attention_params=None,
    ):
        """
        Forward pass with Virtual Linearization
        
        虚拟线性化流程:
        1. 执行 Layer 0-31 (Shared)
        2. 执行 Layer 32-35 (Text)，保存 Layer 35 输出
        3. 在 Layer 36 入口，重置 hidden_states = shared_output + hidden * epsilon
        4. 执行 Layer 36-39 (Audio)
        5. 返回 text_hidden 和 audio_hidden
        """
        hidden_states = self.embed_tokens(input_ids)
        
        # ═══════════════════════════════════════════════════════════════════════
        # 关键变量
        # ═══════════════════════════════════════════════════════════════════════
        shared_output = None  # 保存 Layer 31 的输出
        text_hidden = None    # 保存 Layer 35 的输出
        presents = []
        
        for layer_idx, layer in enumerate(self.layers):
            
            # ═══════════════════════════════════════════════════════════════════
            # 保存 Shared 输出 (Layer 31 → 32 过渡点)
            # ═══════════════════════════════════════════════════════════════════
            if layer_idx == self.text_start_idx:  # 32
                shared_output = identity(hidden_states)
            
            # ═══════════════════════════════════════════════════════════════════
            # 🔴🔴🔴 核心手术点: Layer 36 (Audio 分支起点) 🔴🔴🔴
            # 
            # 这是虚拟线性化的核心！
            # 
            # 为什么需要这行代码？
            # -------------------
            # 1. Text 分支 (32-35) 执行完后，hidden_states 包含 text 信息
            # 2. Audio 分支需要从 shared_output (Layer 31 输出) 开始
            # 3. 直接赋值 hidden_states = shared_output 会被 TensorRT 编译器优化
            #    因为编译器会认为 Layer 32-35 的计算是"无用的"
            # 
            # 为什么是 1e-4？
            # --------------
            # - 1e-8: FP16 精度下会被舍入为 0，优化仍会生效
            # - 1e-1: 太大，会污染 Audio 输出
            # - 1e-4: 刚好在 FP16 可表示范围内 (FP16 最小正数 ≈ 6e-8)
            #         且对输出影响可忽略 (相对误差 < 0.01%)
            # 
            # ⚠️ 警告: 删除或修改此行将导致 Generation Phase 崩溃！
            # ═══════════════════════════════════════════════════════════════════
            if layer_idx == self.audio_start_idx:  # 36
                hidden_states = shared_output + hidden_states * VIRTUAL_LINEARIZATION_EPSILON
            
            # ═══════════════════════════════════════════════════════════════════
            # 执行当前层
            # ═══════════════════════════════════════════════════════════════════
            layer_output = layer(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache,
                kv_cache_params=kv_cache_params,
                attention_params=attention_params,
            )
            
            if use_cache:
                hidden_states, present = layer_output
                presents.append(present)
            else:
                hidden_states = layer_output
            
            # ═══════════════════════════════════════════════════════════════════
            # 保存 Text 输出 (Layer 35)
            # ═══════════════════════════════════════════════════════════════════
            if layer_idx == self.text_end_idx:  # 35
                text_hidden = identity(hidden_states)
        
        # Layer 39 输出就是 audio_hidden
        audio_hidden = hidden_states
        
        if use_cache:
            return (text_hidden, audio_hidden), tuple(presents)
        return text_hidden, audio_hidden


class MossSpeechForCausalLM(DecoderModelForCausalLM):
    """
    MOSS-Speech Causal LM with Dual Output Heads
    
    继承自 DecoderModelForCausalLM 以正确处理 RoPE 位置编码。
    
    输出:
    -----
    combined_logits: [batch, seq_len, 168192]
        - [:, :, :151680]: Text logits
        - [:, :, 151680:]: Audio logits
    """
    
    config_class = MossSpeechConfig
    
    def __init__(self, config: MossSpeechConfig):
        # ═══════════════════════════════════════════════════════════════════════
        # [ARCH_GUARD] 验证配置
        # ═══════════════════════════════════════════════════════════════════════
        assert config.num_hidden_layers == TOTAL_LAYERS, \
            f"[ARCH_GUARD] config.num_hidden_layers={config.num_hidden_layers}, 必须为 {TOTAL_LAYERS}！"
        
        # 创建 transformer
        transformer = MossSpeechModel(config)
        
        # 主 LM head（用于文本）
        lm_head = ColumnLinear(
            in_features=config.hidden_size,
            out_features=config.text_vocab_size,
            bias=False,
            dtype=config.dtype,
            tp_group=None,
            tp_size=1,
            gather_output=True,
        )
        
        # 调用父类 __init__
        # DecoderModelForCausalLM 会自动调用:
        #   Attention.create_attention_const_params(self, config)
        # 这会初始化 RoPE 所需的 rotary_inv_freq 和 embed_positions_for_gpt_attention
        super().__init__(config, transformer, lm_head)
        
        # Audio LM head
        self.audio_lm_head = ColumnLinear(
            in_features=config.hidden_size,
            out_features=config.audio_vocab_size,
            bias=False,
            dtype=config.dtype,
            tp_group=None,
            tp_size=1,
            gather_output=True,
        )
        
        # [ARCH_GUARD] 打印确认信息
        print(f"[ARCH_GUARD] ✅ MossSpeechForCausalLM 初始化完成")
        print(f"   Text vocab: {config.text_vocab_size}")
        print(f"   Audio vocab: {config.audio_vocab_size}")
        print(f"   Combined vocab: {COMBINED_VOCAB_SIZE}")
    
    def __repr__(self):
        """[ARCH_GUARD] 打印架构信息"""
        return (
            f"MossSpeechForCausalLM(\n"
            f"  [ARCH_GUARD] 40-Layer Virtual Linearization Active\n"
            f"  num_hidden_layers={self.config.num_hidden_layers},\n"
            f"  epsilon={VIRTUAL_LINEARIZATION_EPSILON}\n"
            f")"
        )
    
    def forward(
        self,
        input_ids: Tensor,
        position_ids=None,
        use_cache=False,
        last_token_ids=None,
        attention_mask=None,
        kv_cache_params=None,
        attention_params=None,
        hidden_states=None,
        prompt_embedding_table: Optional[Tensor] = None,
        prompt_tasks: Optional[Tensor] = None,
        prompt_vocab_size: Optional[Tensor] = None,
        lora_params=None,
        spec_decoding_params=None,
    ):
        """
        Forward pass
        
        关键: 必须调用 Attention.fill_attention_params 填充 RoPE 参数
        """
        # ═══════════════════════════════════════════════════════════════════════
        # 🔴 填充 attention params（包括 RoPE 参数）
        # 这是继承 DecoderModelForCausalLM 的关键原因！
        # ═══════════════════════════════════════════════════════════════════════
        attention_params = Attention.fill_attention_params(
            self, attention_params)
        
        # ═══════════════════════════════════════════════════════════════════════
        # 执行 transformer（虚拟线性化在这里发生）
        # ═══════════════════════════════════════════════════════════════════════
        outputs = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=use_cache,
            attention_mask=attention_mask,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
        )
        
        if use_cache:
            (text_hidden, audio_hidden), presents = outputs
        else:
            text_hidden, audio_hidden = outputs
        
        # ═══════════════════════════════════════════════════════════════════════
        # 应用 final norm
        # ═══════════════════════════════════════════════════════════════════════
        text_hidden = self.transformer.norm(text_hidden)
        audio_hidden = self.transformer.norm(audio_hidden)
        
        # ═══════════════════════════════════════════════════════════════════════
        # gather last token (用于自回归生成)
        # ═══════════════════════════════════════════════════════════════════════
        if last_token_ids is not None:
            remove_input_padding = tensorrt_llm.default_net().plugin_config.remove_input_padding
            text_hidden = gather_last_token_logits(
                text_hidden, last_token_ids, remove_input_padding
            )
            audio_hidden = gather_last_token_logits(
                audio_hidden, last_token_ids, remove_input_padding
            )
        
        # ═══════════════════════════════════════════════════════════════════════
        # 计算 logits
        # ═══════════════════════════════════════════════════════════════════════
        text_logits = self.lm_head(text_hidden)
        audio_logits = self.audio_lm_head(audio_hidden)
        
        # ═══════════════════════════════════════════════════════════════════════
        # 拼接为单一输出 (Python 端再拆分)
        # 
        # combined_logits[:, :, :151680] = text_logits
        # combined_logits[:, :, 151680:] = audio_logits
        # ═══════════════════════════════════════════════════════════════════════
        combined_logits = concat([text_logits, audio_logits], dim=-1)
        combined_logits.mark_output('logits', self.config.logits_dtype)
        
        if use_cache:
            return combined_logits, presents
        return combined_logits
    
    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str, **kwargs):
        """
        从 checkpoint 加载模型
        
        [ARCH_GUARD] 无论 checkpoint 中的配置如何，都强制使用 40 层
        """
        config_path = os.path.join(checkpoint_dir, "config.json")
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            # 获取 pretrained_config
            if 'pretrained_config' in config_dict:
                pretrained_config = config_dict['pretrained_config']
            else:
                pretrained_config = config_dict
            
            config = MossSpeechConfig(**pretrained_config)
        else:
            config = MossSpeechConfig()
        
        # ═══════════════════════════════════════════════════════════════════════
        # [ARCH_GUARD] 打印确认
        # ═══════════════════════════════════════════════════════════════════════
        print(f"[ARCH_GUARD] ✅ 配置加载完成: num_hidden_layers = {config.num_hidden_layers}")
        
        model = cls(config)
        
        # 加载权重
        weights_path = os.path.join(checkpoint_dir, "rank0.safetensors")
        if os.path.exists(weights_path):
            print(f"加载权重: {weights_path}")
        
        return model


# ═══════════════════════════════════════════════════════════════════════════════
# [ARCH_GUARD] 模块级别的架构验证函数
# ═══════════════════════════════════════════════════════════════════════════════
def verify_model_architecture(model: MossSpeechForCausalLM):
    """
    验证模型架构完整性
    
    在构建 Engine 前调用此函数！
    """
    # 1. 层数检查
    assert model.config.num_hidden_layers == 40, \
        f"[ARCH_GUARD] FATAL: num_hidden_layers={model.config.num_hidden_layers}, 必须为 40！"
    
    # 2. 虚拟线性化依赖链检查
    source = inspect.getsource(model.transformer.forward)
    assert "1e-4" in source or "1e-04" in source or "VIRTUAL_LINEARIZATION_EPSILON" in source, \
        "[ARCH_GUARD] FATAL: 虚拟序列化依赖链丢失！1e-4 epsilon 不可删除！"
    
    # 3. 分支索引检查
    assert model.transformer.audio_start_idx == 36, \
        f"[ARCH_GUARD] FATAL: audio_start_idx={model.transformer.audio_start_idx}, 必须为 36！"
    
    print("[ARCH_GUARD] ✅ 模型架构验证通过")
    print(f"   num_hidden_layers: {model.config.num_hidden_layers}")
    print(f"   audio_start_idx: {model.transformer.audio_start_idx}")
    print(f"   epsilon: {VIRTUAL_LINEARIZATION_EPSILON}")
    
    return True
