# CosyVoice2 性能诊断报告

**日期**: 2026-01-12  
**测试环境**: RunPod RTX 4090 (24GB)  
**模型**: CosyVoice2 0.5B (fp16=True)  

---

## 📊 测试结果概览

| 指标 | 测量值 | 目标值 | 状态 |
|------|--------|--------|------|
| **TTFA (流式)** | 16,315 ms | < 300 ms | ❌ 超标 54x |
| **RTF (非流式)** | 6.48 | < 0.3 | ❌ 超标 22x |
| **RTF (流式)** | 7.80 | < 0.3 | ❌ 超标 26x |

**结论**: 当前配置下 CosyVoice2 **完全无法实时使用**

---

## 🔍 当前模型配置

```python
# mouth_cosyvoice2.py 第 147 行
self.model = CosyVoice2(MODEL_DIR, fp16=True)
```

| 配置项 | 当前值 | 推荐值 | 说明 |
|--------|--------|--------|------|
| `fp16` | ✅ True | True | 半精度推理 |
| `load_jit` | ❌ False | True | JIT 编译优化 |
| `load_trt` | ❌ False | True | TensorRT 加速 |
| `load_vllm` | ❌ False | True | VLLM LLM 优化 |

---

## 🐌 TTFA 延迟瓶颈分析

### Zero-shot 推理流程

```
请求到达 ──┬──► [1] 文本标准化 (text_normalize)
           │         ~10ms
           │
           ├──► [2] 前端处理 (frontend_zero_shot)  ◄── 主要瓶颈 1
           │    ├── _extract_text_token (tokenize)
           │    │         ~50ms
           │    ├── _extract_speech_token (whisper + ONNX)  ◄── 瓶颈
           │    │         ~2000-3000ms (处理 3.5s 参考音频)
           │    ├── _extract_spk_embedding (CAM++ ONNX)
           │    │         ~500ms
           │    └── _extract_speech_feat (特征提取)
           │              ~100ms
           │
           └──► [3] 模型推理 (model.tts)  ◄── 主要瓶颈 2
                ├── LLM 生成 speech tokens  ◄── 最大瓶颈
                │         ~8000-12000ms (无 VLLM 优化)
                ├── Flow 模型推理
                │         ~2000ms (无 TensorRT)
                └── HiFi-GAN 声码器
                          ~500ms
                          
总计: ~16000ms (16秒)
```

### 瓶颈详细分析

#### 瓶颈 1: Speech Tokenizer (ONNX)

```python
# frontend.py 第 72-79 行
feat = whisper.log_mel_spectrogram(speech, n_mels=128)
speech_token = self.speech_tokenizer_session.run(...)
```

- 使用 Whisper 提取 Mel 特征
- ONNX 运行 speech tokenizer
- **问题**: 每次请求都要处理参考音频，无缓存机制

#### 瓶颈 2: LLM 推理 (无 VLLM)

```python
# model.py 第 115-131 行
token_generator = self.llm.inference(...)
for i in token_generator:
    self.tts_speech_token_dict[uuid].append(i)
```

- 原生 PyTorch LLM 推理
- 无 KV-Cache 复用
- 无批量优化
- **问题**: 未使用 VLLM 优化，推理极慢

#### 瓶颈 3: Flow 解码器 (无 TensorRT)

```python
# model.py 第 152-159 行  
tts_mel, _ = self.flow.inference(...)
```

- 原生 PyTorch Flow 模型
- **问题**: 未使用 TensorRT 加速

---

## 🔧 存储问题

```bash
$ df -h /workspace/models
Filesystem                   Size  Used Avail Use% Mounted on
mfs#us-il-1.runpod.net:9421  503T  319T  185T  64% /workspace
```

**问题**: 模型存储在 **网络文件系统 (NFS/MFS)** 上！

| 操作 | 预估延迟 |
|------|---------|
| 本地 NVMe SSD | ~0.1ms |
| 网络存储 NFS | ~10-50ms |
| 模型权重加载 | +500-2000ms 首次延迟 |

---

## 📁 可用优化文件状态

```
/workspace/models/CosyVoice/pretrained_models/iic/CosyVoice2-0___5B/
├── flow.encoder.fp16.zip   ✅ 存在 (JIT, 未使用)
├── flow.encoder.fp32.zip   ✅ 存在 (JIT, 未使用)
├── *.plan                  ❌ 不存在 (TensorRT)
└── vllm/                   ❌ 不存在 (VLLM)
```

---

## 💡 优化建议

### 优先级 1: 启用 JIT 编译 (预计提升 20-30%)

```python
# 修改 mouth_cosyvoice2.py
self.model = CosyVoice2(MODEL_DIR, fp16=True, load_jit=True)
```

### 优先级 2: 生成并使用 TensorRT (预计提升 50-70%)

```bash
# 需要先生成 TensorRT plan 文件
# CosyVoice 会在首次 load_trt=True 时自动生成

self.model = CosyVoice2(MODEL_DIR, fp16=True, load_jit=True, load_trt=True)
```

### 优先级 3: 配置 VLLM (预计提升 80-90%)

```bash
# 需要导出 VLLM 格式
# 参考 vllm_example.py

self.model = CosyVoice2(MODEL_DIR, fp16=True, load_jit=True, load_trt=True, load_vllm=True)
```

### 优先级 4: 模型预热 + 参考音频缓存

```python
# 预热 Zero-shot 推理
warmup_prompt_wav = "/path/to/prompt.wav"
warmup_prompt_text = "预热文本"
for _ in self.model.inference_zero_shot("测试", warmup_prompt_text, warmup_prompt_wav, stream=False):
    pass

# 缓存常用说话人
self.model.add_zero_shot_spk(prompt_text, prompt_wav, "cached_speaker_id")
```

### 优先级 5: 将模型移至本地 SSD

```bash
# 复制到本地存储
cp -r /workspace/models/CosyVoice/pretrained_models /tmp/cosyvoice_models/
# 修改 MODEL_DIR 指向本地路径
```

---

## 📈 预期性能提升

| 优化方案 | 预计 TTFA | 预计 RTF |
|----------|-----------|----------|
| 当前配置 (baseline) | 16,000 ms | 6.5 |
| + JIT | ~12,000 ms | ~5.0 |
| + TensorRT | ~5,000 ms | ~2.0 |
| + VLLM | ~1,500 ms | ~0.5 |
| + 预热 + 缓存 | ~500 ms | ~0.3 |
| + 本地 SSD | ~300 ms | ~0.2 |

---

## 🎯 结论

CosyVoice2 在当前配置下性能极差的**根本原因**:

1. ❌ **未使用任何优化**: JIT、TensorRT、VLLM 全部关闭
2. ❌ **未预热**: 跳过预热导致首次推理冷启动
3. ❌ **网络存储**: 模型在 NFS 上，I/O 延迟高
4. ❌ **无缓存**: 每次请求都重新处理参考音频

**建议**: 如果需要实时 TTS，考虑:
- 方案 A: 全面优化 CosyVoice2 (启用 JIT + TRT + VLLM)
- 方案 B: 切换到其他更快的 TTS 引擎 (如 VoxCPM)

---

*报告生成时间: 2026-01-12 00:50 UTC*


