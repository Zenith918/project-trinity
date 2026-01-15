# MOSS-Speech TensorRT-LLM 优化实施报告

**日期**: 2026-01-14  
**目标**: 将 MOSS-Speech 9.1B 模型部署到 TensorRT-LLM，实现 RTF < 1  
**环境**: RunPod RTX 4090 (24GB VRAM), CUDA 12.4

---

## 一、执行摘要

| 阶段 | 状态 | 关键产出 |
|-----|-----|---------|
| 环境配置 | ✅ | tensorrt-llm 0.13.0 + tensorrt 10.3.0 |
| 模型下载 | ✅ | /workspace/models/MOSS-Speech (17GB) |
| 架构分析 | ✅ | 确认 Layer-Splitting 而非 Flow-matching |
| 权重转换 | ✅ | /workspace/models/MOSS-Speech-TRTLLM (14GB checkpoint) |
| Engine 构建 | ✅ | /workspace/models/MOSS-Speech-TRTLLM-Engine (6.3GB) |
| BigVGAN 准备 | ✅ | /workspace/models/BigVGAN (429MB weights) |

---

## 二、详细执行过程

### 2.1 TensorRT-LLM 环境配置

#### 执行步骤

```bash
# 1. 创建独立虚拟环境
python -m venv /workspace/envs/trtllm
source /workspace/envs/trtllm/bin/activate

# 2. 安装 TensorRT 基础库
pip install tensorrt==10.3.0 --extra-index-url https://pypi.nvidia.com

# 3. 安装 TensorRT-LLM (CUDA 12.4 兼容版本)
pip install tensorrt-llm==0.13.0 --extra-index-url https://pypi.nvidia.com

# 4. 修复 cuda-python 版本冲突
pip uninstall cuda-python cuda-bindings -y
pip install 'cuda-python==12.9.5'
```

#### 遇到的问题 #1: `ImportError: cannot import name 'cudart' from 'cuda'`

**原因**: `cuda-python` 默认安装了 13.x 版本，与 CUDA 12.4 不兼容

**解决方案**:
```bash
pip install 'cuda-python<13'
```

#### 遇到的问题 #2: `AttributeError: module 'pynvml' has no attribute '__version__'`

**原因**: 新版 `nvidia-ml-py` 移除了 `__version__` 属性，但 TensorRT-LLM 代码中有版本检查

**解决方案**: 手动修补两个文件

**文件 1**: `/workspace/envs/trtllm/lib/python3.10/site-packages/tensorrt_llm/profiler.py`
```python
# 修改前 (第 30 行左右)
if pynvml.__version__ < '11.5.0' or driver_version < '526':

# 修改后
# if pynvml.__version__ < '11.5.0' or driver_version < '526':  # DISABLED
```

**文件 2**: `/workspace/envs/trtllm/lib/python3.10/site-packages/tensorrt_llm/auto_parallel/cluster_info.py`
```python
# 修改前 (第 484、509 行)
if pynvml.__version__ < '11.5.0':

# 修改后
if False:  # pynvml.__version__ check disabled
```

---

### 2.2 MOSS-Speech 模型下载与分析

#### 执行步骤

```python
from transformers import AutoModel, AutoTokenizer
import torch

model = AutoModel.from_pretrained(
    "OpenMOSS-Team/MOSS-Speech",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    cache_dir="/workspace/models/MOSS-Speech"
)
```

#### 架构发现

**关键发现**: MOSS-Speech **不是** Flow-matching 架构，而是 **Layer-Splitting Transformer**！

```
┌─────────────────────────────────────────┐
│        MOSS-Speech 架构                  │
├─────────────────────────────────────────┤
│                                         │
│  ┌───────────────────────────────────┐  │
│  │     embed_tokens + audio_embed    │  │
│  └───────────────┬───────────────────┘  │
│                  │                      │
│  ┌───────────────▼───────────────────┐  │
│  │        shared_block (32层)        │  │
│  │      Qwen2 Backbone               │  │
│  └───────────────┬───────────────────┘  │
│                  │                      │
│         ┌───────┴───────┐              │
│         │               │              │
│  ┌──────▼──────┐ ┌──────▼──────┐       │
│  │ text_block  │ │ audio_block │       │
│  │  (4层)      │ │   (4层)     │       │
│  └──────┬──────┘ └──────┬──────┘       │
│         │               │              │
│  ┌──────▼──────┐ ┌──────▼──────┐       │
│  │  text_head  │ │ audio_head  │       │
│  └─────────────┘ └─────────────┘       │
│                                         │
└─────────────────────────────────────────┘
```

**模型参数统计**:
| 组件 | 参数量 | 说明 |
|------|--------|------|
| shared_block | 6.17B | 32 层共享 Transformer |
| text_block | 771.8M | 4 层文本专用 |
| audio_block | 771.8M | 4 层音频专用 |
| embed_tokens | 621.3M | 文本嵌入 |
| audio_embed | 67.6M | 音频嵌入 |
| **总计** | **9.10B** | - |

---

### 2.3 权重转换 (HuggingFace → TensorRT-LLM)

#### 遇到的问题 #3: `KeyError: 'MossSpeechForCausalLM'`

**原因**: TensorRT-LLM 的 `MODEL_MAP` 不包含自定义架构，只支持官方模型（Qwen、Llama 等）

**解决方案**: 将 MOSS-Speech 的 `shared_block` (32层) 映射为 `Qwen2ForCausalLM` 格式

#### 遇到的问题 #4: 权重键名不匹配

**原因**: 我们第一版 `convert.py` 没有生成 TRT-LLM Qwen 期望的所有权重

**缺失的权重**:
- `transformer.layers.{i}.attention.qkv.bias` (32层 × 每层)
- `transformer.ln_f.weight`

**解决方案**: 重写 `convert_v2.py`

```python
# convert_v2.py 关键代码

def convert_moss_to_qwen2_trtllm(hf_weights, num_layers=32):
    trtllm_weights = {}
    
    for layer_idx in range(num_layers):
        prefix = f"model.shared_block.layers.{layer_idx}"
        trt_prefix = f"transformer.layers.{layer_idx}"
        
        # 合并 Q、K、V 为 QKV
        q = hf_weights[f"{prefix}.self_attn.q_proj.weight"]
        k = hf_weights[f"{prefix}.self_attn.k_proj.weight"]
        v = hf_weights[f"{prefix}.self_attn.v_proj.weight"]
        qkv = torch.cat([q, k, v], dim=0)
        trtllm_weights[f"{trt_prefix}.attention.qkv.weight"] = qkv
        
        # 添加零 bias (MOSS-Speech 没有 bias，但 TRT-LLM 需要)
        qkv_bias = torch.zeros(qkv.shape[0], dtype=dtype)
        trtllm_weights[f"{trt_prefix}.attention.qkv.bias"] = qkv_bias
        
        # ... 其他权重映射
    
    # Final LayerNorm
    trtllm_weights["transformer.ln_f.weight"] = hf_weights["model.text_norm.weight"]
    
    return trtllm_weights
```

**权重映射表**:

| MOSS-Speech 键名 | TRT-LLM 键名 |
|-----------------|-------------|
| `model.shared_block.layers.{i}.self_attn.q_proj.weight` | `transformer.layers.{i}.attention.qkv.weight` (合并) |
| `model.shared_block.layers.{i}.self_attn.k_proj.weight` | ↑ |
| `model.shared_block.layers.{i}.self_attn.v_proj.weight` | ↑ |
| `model.shared_block.layers.{i}.self_attn.o_proj.weight` | `transformer.layers.{i}.attention.dense.weight` |
| `model.shared_block.layers.{i}.mlp.gate_proj.weight` | `transformer.layers.{i}.mlp.gate.weight` |
| `model.shared_block.layers.{i}.mlp.up_proj.weight` | `transformer.layers.{i}.mlp.fc.weight` |
| `model.shared_block.layers.{i}.mlp.down_proj.weight` | `transformer.layers.{i}.mlp.proj.weight` |
| `model.shared_block.layers.{i}.input_layernorm.weight` | `transformer.layers.{i}.input_layernorm.weight` |
| `model.shared_block.layers.{i}.post_attention_layernorm.weight` | `transformer.layers.{i}.post_layernorm.weight` |
| `model.text_norm.weight` | `transformer.ln_f.weight` |
| `model.embed_tokens.weight` | `transformer.vocab_embedding.weight` |
| `text_lm_head.weight` | `lm_head.weight` |

#### 转换结果

```
/workspace/models/MOSS-Speech-TRTLLM/
├── config.json          # TRT-LLM 配置
└── rank0.safetensors    # 14GB 权重 (259 个 tensor)
```

**config.json 内容**:
```json
{
  "architecture": "Qwen2ForCausalLM",
  "dtype": "float16",
  "num_hidden_layers": 32,
  "hidden_size": 4096,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "vocab_size": 151680,
  "qwen_type": "qwen2",
  "bias": true
}
```

---

### 2.4 TensorRT Engine 构建

#### 执行命令

```bash
trtllm-build \
    --checkpoint_dir /workspace/models/MOSS-Speech-TRTLLM \
    --output_dir /workspace/models/MOSS-Speech-TRTLLM-Engine \
    --gemm_plugin float16 \
    --kv_cache_type paged \
    --context_fmha enable \
    --max_batch_size 1 \
    --max_input_len 2048 \
    --max_seq_len 6144 \
    --remove_input_padding enable
```

#### 遇到的问题 #5: quantization 格式不兼容

**错误**: `TypeError: QuantConfig.__init__() got an unexpected keyword argument 'use_fp8'`

**原因**: 我们自定义的 config.json 中 `quantization` 格式与 TRT-LLM 期望不符

**解决方案**:
```python
# 修改前
"quantization": {
    "use_fp8": false,
    "use_int8_weight_only": false
}

# 修改后
"quantization": {
    "quant_algo": null,
    "kv_cache_quant_algo": null
}
```

#### 遇到的问题 #6: 缺少 `qwen_type` 配置

**错误**: `AttributeError: 'QWenConfig' object has no attribute 'qwen_type'`

**解决方案**: 在 config.json 添加
```json
"qwen_type": "qwen2"
```

#### 构建结果

```
/workspace/models/MOSS-Speech-TRTLLM-Engine/
├── config.json      # 6 KB (完整构建配置)
└── rank0.engine     # 6.3 GB (TensorRT Engine)
```

**构建统计**:
| 指标 | 值 |
|-----|-----|
| Engine 大小 | 6.3 GB |
| 构建时间 | 107.8 秒 |
| 峰值 GPU 显存 | 14,147 MiB |
| 峰值 CPU 内存 | 37,196 MiB |
| 权重内存 | 14.8 GB |

**启用的优化**:
- ✅ PagedAttention (`kv_cache_type: PAGED`)
- ✅ Context FMHA (`context_fmha: true`)
- ✅ Remove Input Padding (`remove_input_padding: true`)
- ✅ Fused MLP (`use_fused_mlp: true`)
- ✅ XQA (`enable_xqa: true`)

---

### 2.5 BigVGAN-v2 准备

#### 执行步骤

```bash
# 1. 克隆仓库
git clone https://github.com/NVIDIA/BigVGAN.git /workspace/models/BigVGAN

# 2. 安装依赖
pip install ninja matplotlib librosa soundfile

# 3. 下载预训练权重
python -c "
from bigvgan import BigVGAN
model = BigVGAN.from_pretrained('nvidia/bigvgan_v2_22khz_80band_256x')
torch.save({'generator': model.state_dict()}, 'bigvgan_v2_22khz_80band_256x.pt')
"
```

#### 结果

```
/workspace/models/BigVGAN/
├── bigvgan.py
├── bigvgan_v2_22khz_80band_256x.pt  # 429 MB
└── ...
```

---

## 三、当前限制与待解决问题

### 3.1 架构限制

当前转换只包含 **shared_block (32层)**，未包含：
- `text_block` (4层) - 文本专用层
- `audio_block` (4层) - 音频专用层
- `audio_embed` - 音频嵌入
- `audio_lm_head` - 音频输出头

**原因**: TRT-LLM 标准 Qwen 架构不支持双分支结构

**影响**: 当前 Engine 只能用于文本生成，完整的 Speech-to-Speech 需要额外集成

### 3.2 待验证项目

| 项目 | 状态 | 说明 |
|-----|-----|-----|
| RTF 基准测试 | ⏳ | 需运行 benchmark.py |
| BigVGAN 集成 | ⏳ | CUDA Kernel 编译 |
| 流式推理 | ⏳ | StreamingBuffer 测试 |
| 双分支支持 | ⏳ | 需要 TRT-LLM Python Model API |

---

## 四、文件清单

### 新增/修改的文件

```
/workspace/project-trinity/project-trinity/server/trtllm_moss/
├── __init__.py              # 模块入口
├── model.py                 # TRT-LLM 自定义模型定义
├── convert.py               # 原始转换脚本 (弃用)
├── convert_v2.py            # 修复版转换脚本 ✅
├── inference.py             # 推理接口
├── streaming_inference.py   # 流式推理
├── vocoder.py               # BigVGAN 集成
└── build_engine.sh          # 构建脚本

/workspace/models/
├── MOSS-Speech/             # 原始 HuggingFace 模型 (17GB)
├── MOSS-Speech-TRTLLM/      # TRT-LLM checkpoint (14GB)
├── MOSS-Speech-TRTLLM-Engine/  # TensorRT Engine (6.3GB)
└── BigVGAN/                 # BigVGAN vocoder (429MB)
```

### 修改的 TRT-LLM 源文件

```
/workspace/envs/trtllm/lib/python3.10/site-packages/tensorrt_llm/
├── profiler.py              # pynvml 版本检查禁用
└── auto_parallel/
    └── cluster_info.py      # pynvml 版本检查禁用
```

---

## 五、下一步建议

1. **性能测试**: 运行 RTF 基准测试，验证 TRT-LLM 加速效果
2. **双分支支持**: 使用 TRT-LLM Python Model API 实现完整 MOSS-Speech 架构
3. **BigVGAN 集成**: 编译 CUDA Kernel，测试 vocoder 性能
4. **端到端测试**: 文本输入 → 音频输出全链路测试

---

## 六、参考命令

### 验证 Engine

```python
from tensorrt_llm.runtime import ModelRunner

runner = ModelRunner.from_dir("/workspace/models/MOSS-Speech-TRTLLM-Engine")
print(f"Engine loaded: {runner}")
```

### 运行推理

```python
output = runner.generate(
    input_ids,
    max_new_tokens=100,
    end_id=tokenizer.eos_token_id,
)
```

---

---

## 七、研究员 P0 指示执行进展 (18:40 更新)

### 7.1 完整权重转换 ✅

成功转换 **完整架构 (32+4+4)**：

```
/workspace/models/MOSS-Speech-TRTLLM-Full/
├── config.json          # 完整配置
└── rank0.safetensors    # 17 GB (326 tensors)
```

**权重统计**:
| 组件 | Tensors | 说明 |
|-----|---------|-----|
| shared_block | 256 | 32层 × 8 tensors/层 |
| text_block | 32 | 4层 × 8 tensors/层 |
| audio_block | 32 | 4层 × 8 tensors/层 |
| embed_tokens | 1 | 文本嵌入 (151680 × 4096) |
| audio_embed | 1 | 音频嵌入 (16512 × 4096) |
| text_norm | 1 | 文本 RMSNorm |
| audio_norm | 1 | 音频 RMSNorm |
| text_lm_head | 1 | 文本输出头 (4096 → 151680) |
| audio_lm_head | 1 | 音频输出头 (4096 → 16512) |

### 7.2 Engine 构建挑战

**问题**: TRT-LLM 的 `trtllm-build` 命令行工具不支持自定义双分支架构

**解决方案选项**:

| 方案 | 优点 | 缺点 |
|-----|-----|-----|
| **A. PyTorch → ONNX → TRT** | 通用，稳定 | 丢失 PagedAttention |
| **B. TRT-LLM Python Model API** | 原生支持 KV Cache | 需要大量自定义代码 |
| **C. vLLM** | 自定义模型支持好 | 需要重新适配 |
| **D. 分块构建** | 保留 TRT-LLM 优化 | 运行时需要串联多个 Engine |

**当前选择**: 方案 D (分块构建)
- shared_block → 使用现有 Qwen2 Engine (6.3GB, 已构建)
- text/audio branches → 使用 ONNX → TRT

### 7.3 新增文件

```
server/trtllm_moss/
├── moss_speech_model.py   # 完整 TRT-LLM 模型定义
├── convert_full.py        # 完整权重转换 (32+4+4)
└── build_full_engine.py   # 完整 Engine 构建脚本
```

---

**报告人**: AI Assistant  
**审核**: 待研究员确认

