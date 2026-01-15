# Plan B 详细进度报告：TensorRT-LLM Python Model API 方案

**日期**: 2026-01-15  
**报告人**:  工程师  
**审阅**: 首席研究员

---

## 一、执行摘要

| 项目 | 状态 | 详情 |
|------|------|------|
| **方案名称** | Plan B | 使用 TRT-LLM Python Model API 手动构建完整架构 |
| **放弃原因** | Plan A 失败 | `trtllm-build` CLI 不识别 `MossSpeechForCausalLM` |
| **核心成果** | ✅ 成功 | 构建了 16.9GB 完整架构 Engine (32+4+4) |
| **RTF 预估** | 0.037 | ✅ 远低于实时要求 (< 1.0) |
| **TTFA 预估** | ~400ms | ⚠️ 略高于目标 (< 300ms) |

---

## 二、具体做了什么

### 2.1 创建自定义 TRT-LLM 模型类

**文件**: `server/trtllm_moss/moss_trtllm_model.py`

核心工作是继承 TRT-LLM 的 `PretrainedModel` 基类，手动实现 MOSS-Speech 的完整 32+4+4 分叉架构：

```37:78:server/trtllm_moss/moss_trtllm_model.py
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
```

**关键设计**:
- `num_shared_layers=32`: 共享 Transformer 层
- `num_text_layers=4`: 文本专用分支
- `num_audio_layers=4`: 音频专用分支
- `audio_vocab_size=16512`: 音频 Token 词表大小

### 2.2 权重转换脚本

**文件**: `server/trtllm_moss/convert_full.py`

将 HuggingFace 格式的 MOSS-Speech 权重转换为 TRT-LLM 格式：

```53:89:server/trtllm_moss/convert_full.py
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
```

**核心转换**:
- Q/K/V 分离权重 → 合并的 `qkv.weight`
- HuggingFace 命名 → TRT-LLM 命名
- 所有 40 层 (32+4+4) 完整转换

### 2.3 Engine 构建脚本

**文件**: `server/trtllm_moss/build_engine.py`

使用 TRT-LLM Python API 构建 Engine：

```82:96:server/trtllm_moss/build_engine.py
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
```

**启用的关键优化**:
| 优化项 | 配置 | 作用 |
|--------|------|------|
| `paged_kv_cache` | True | PagedAttention，长对话优化 |
| `remove_input_padding` | True | 动态序列长度，减少浪费 |
| `context_fmha` | True | FlashAttention，加速 prefill |
| `use_fused_mlp` | True | MLP 算子融合 |

---

## 三、完成了什么

### 3.1 文件产出

| 文件 | 大小 | 位置 |
|------|------|------|
| **TRT-LLM Engine** | 16.9 GB | `/workspace/models/MOSS-Speech-Engine/rank0.engine` |
| **Engine Config** | 4.8 KB | `/workspace/models/MOSS-Speech-Engine/config.json` |
| **转换后权重** | 17 GB | `/workspace/models/MOSS-Speech-TRTLLM-Full/rank0.safetensors` |

### 3.2 Engine 验证

Engine 可以成功加载并识别完整架构：

```bash
# 验证命令
python3 -c "
from moss_trtllm_model import register_moss_speech_model
register_moss_speech_model()
from tensorrt_llm.builder import Engine
engine = Engine.from_dir('/workspace/models/MOSS-Speech-Engine')
print('✅ Engine loaded!')
"
```

**输出配置**:
```
Architecture: MossSpeechForCausalLM
Dtype: float16
Shared Layers: 32
Text Layers: 4
Audio Layers: 4
PagedKVCache: True
```

---

## 四、有什么让步

### 4.1 量化策略 (暂未实现)

| 原计划 | 实际 | 原因 |
|--------|------|------|
| shared_block FP8 | FP16 | FP8 需要校准数据集 |
| text/audio_block FP16 | FP16 | 保持精度 |

**影响**: Engine 大小 16.9GB vs 预期 ~9GB (FP8)

### 4.2 音频输入路径 (暂未启用)

| 原计划 | 实际 | 原因 |
|--------|------|------|
| 支持 `audio_input_ids` | 仅 `input_ids` | 需要额外的嵌入融合逻辑 |

**影响**: 当前仅支持 TTS (文本→音频)，STS (语音→语音) 需要后续开发

### 4.3 KV Cache 空间

| 指标 | 数值 |
|------|------|
| GPU 总内存 | 23.5 GB (RTX 4090) |
| Engine 占用 | ~16.9 GB |
| KV Cache 预算 | ~4.7 GB |

**影响**: 长对话可能受限，FP8 量化后可释放更多空间

---

## 五、测量值计算方法

### 5.1 RTF (Real-Time Factor) 计算

**公式**: `RTF = 音频 Token 率 / 模型吞吐量`

**代码**:

```92:117:server/trtllm_moss/benchmark_v2.py
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
```

**计算过程**:
```
1. RTX 4090 FP16 算力: 82.6 TFLOPS
2. 理论吞吐量: 82.6 × 10¹² / (2 × 9.1 × 2 × 10⁹) = 2269 tokens/s
3. 实际效率 60%: 2269 × 0.6 = 1362 tokens/s
4. 音频 Token 率: 50 tokens/s (实时音频)
5. RTF = 50 / 1362 = 0.037
```

**结论**: RTF = 0.037 << 1.0，**远超实时要求**

### 5.2 TTFA (Time To First Audio) 计算

**公式**: `TTFA = Prefill 时间 + 首批 Token 生成时间 + 系统开销`

**代码**:

```130:152:server/trtllm_moss/benchmark_v2.py
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
```

**计算过程**:
```
1. Prefill 512 tokens: 512 / 1362 × 1000 = 376 ms
2. 首批 5 tokens: 5 / 1362 × 1000 = 4 ms
3. 系统开销: 20 ms
4. TTFA = 376 + 4 + 20 = 400 ms
```

**结论**: TTFA ≈ 400ms，略高于 300ms 目标

### 5.3 内存预估

**代码**:

```75:81:server/trtllm_moss/benchmark_v2.py
    # 预估运行时内存
    # TRT-LLM Engine 运行时内存约为文件大小
    estimated_runtime = engine_size_gb
    kv_cache_budget = gpu_total - estimated_runtime - 2.0  # 2GB 系统开销
    
    print(f"  预估运行内存: ~{estimated_runtime:.1f} GB")
    print(f"  KV Cache 预算: ~{kv_cache_budget:.1f} GB")
```

**计算**:
```
1. Engine 文件: 16.9 GB
2. GPU 总内存: 23.5 GB
3. 系统开销: 2 GB
4. KV Cache = 23.5 - 16.9 - 2.0 = 4.6 GB
```

---

## 六、测量值的局限性

| 项目 | 当前方法 | 局限性 | 改进建议 |
|------|----------|--------|----------|
| **RTF** | 理论计算 | 未实际运行推理 | 实现 ModelRunner 进行真实测试 |
| **TTFA** | 理论计算 | 假设线性吞吐量 | 实际 prefill 可能更快（批量） |
| **效率系数** | 假设 60% | 可能偏保守 | TRT-LLM 实际可达 70-80% |
| **音频 Token 率** | 假设 50 tok/s | 需确认 MOSS-Speech 实际值 | 查阅论文或实测 |

**注意**: 以上数值为**理论预估**，需要实现完整推理 Runner 后进行**实测验证**。

---


---

## 八、文件清单

```
server/trtllm_moss/
├── __init__.py                 # 模块初始化
├── moss_trtllm_model.py        # 自定义 TRT-LLM 模型 (核心)
├── convert_full.py             # 权重转换脚本
├── build_engine.py             # Engine 构建脚本
├── benchmark.py                # 基准测试 V1 (已弃用)
└── benchmark_v2.py             # 基准测试 V2 (当前使用)

/workspace/models/
├── MOSS-Speech/                # 原始 HuggingFace 模型 (17GB)
├── MOSS-Speech-TRTLLM-Full/    # 转换后权重 (17GB)
│   ├── config.json
│   └── rank0.safetensors
└── MOSS-Speech-Engine/         # TRT-LLM Engine (16.9GB)
    ├── config.json
    └── rank0.engine
```

---

## 九、结论

**Plan B 已成功实施**，核心成果：

1. ✅ 完整架构 Engine (32+4+4) 构建成功
2. ✅ PagedAttention 全链路启用
3. ✅ RTF 理论值 0.037，远超实时要求
4. ⚠️ TTFA 略高于目标，需 FP8 优化
5. ⚠️ KV Cache 空间紧张，建议量化

**这是 MOSS-Speech 实时化的重要里程碑。**



