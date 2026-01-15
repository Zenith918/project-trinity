# MOSS-Speech 迁移工程计划书

> 基于 Project Trinity 当前架构的可执行改动方案
> 编制日期：2026-01-13

---

## 📋 执行摘要

### 核心目标
将语音交互从**级联 TTS 架构**升级为**端到端 STS（Speech-to-Speech）架构**，目标延迟从当前的 **~1.2秒** 降至 **<300ms**。

### 当前架构 vs 目标架构

| 维度 | 当前架构 (CosyVoice3) | 目标架构 (MOSS-Speech) |
|------|----------------------|------------------------|
| 类型 | TTS (Text-to-Speech) | STS (Speech-to-Speech) |
| 流程 | ASR → Brain → TTS | 端到端语音直出 |
| TTFA | ~1200ms | <300ms |
| 情感感知 | 需显式标注 | 自动从语音中提取 |
| 打断能力 | 需额外实现 | 原生支持 (Barge-in) |

---

## ⚠️ 关键雷点与风险清单

### 🔴 高风险 - 必须解决

| 编号 | 风险点 | 影响 | 缓解措施 |
|------|--------|------|----------|
| R1 | **显存冲突** | 当前 CosyVoice3+VLLM 占用 ~9GB，MOSS-Speech 7B FP16 需要 ~15GB，24GB 不足以同时运行 | 需要**二选一**或**时分复用** |
| R2 | **CUDA 版本不匹配** | MOSS-Speech 要求 CUDA 12.1，当前编译工具是 12.4 | PyTorch 已用 cu121 编译，**问题不大** |
| R3 | **模型成熟度** | MOSS-Speech 是学术新模型，可能存在未知 Bug | 设置**回退机制**到 CosyVoice3 |
| R4 | **16kHz 音质限制** | 16kHz 仅覆盖人声主频段，音乐/环境音表现差 | 仅用于对话场景，播报场景保留 CosyVoice |

### 🟡 中风险 - 需要关注

| 编号 | 风险点 | 影响 | 缓解措施 |
|------|--------|------|----------|
| R5 | **依赖版本冲突** | transformers 4.37 vs 当前版本可能冲突 | 使用 Docker 隔离 |
| R6 | **Flash Attention 兼容** | FA2 需要特定编译参数 | 已安装 2.8.3 ✅ |
| R7 | **XY-Tokenizer 学习成本** | 新的音频编解码范式 | 参考官方 Demo 逐步迁移 |
| R8 | **声学幻觉** | 模型可能产生奇怪的笑声/叹气 | 降低 temperature 到 0.4-0.6 |

### 🟢 低风险 - 可控

| 编号 | 风险点 | 影响 | 说明 |
|------|--------|------|------|
| R9 | 中文能力 | MOSS 基于 Qwen 基座，中文能力强 | 优势点 |
| R10 | WebSocket 改造 | 需要全双工通信 | 当前已有 `/tts/ws` 接口，可复用 |

---

## 🏗️ 当前环境现状

```
硬件: NVIDIA RTX 4090 24GB
CUDA: 12.4 (编译工具) / 12.1 (PyTorch)
PyTorch: 2.4.1+cu121
Flash Attention: 2.8.3 ✅
显存使用: ~9GB (CosyVoice3 运行时)
剩余显存: ~15GB
```

### 当前服务架构
```
                    ┌─────────────┐
   用户语音 ──────▶ │   Ear ASR   │ ──────▶ 文本
                    └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
      文本 ────────▶│    Brain    │ ──────▶ 回复文本
                    │   (LLM)     │
                    └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
   回复文本 ───────▶│   Mouth     │ ──────▶ 音频
                    │ (CosyVoice) │
                    └─────────────┘
```

### 目标服务架构 (MOSS-Speech)
```
                    ┌─────────────────────────┐
   用户语音 ──────▶ │      MOSS-Speech        │ ──────▶ 回复音频
     (流式)        │  (端到端 STS 模型)       │    (流式)
                    │                         │
                    │  ┌─────────────────┐   │
                    │  │ Speech Encoder  │   │ ◀── 直接理解语音情感
                    │  └────────┬────────┘   │
                    │           ▼            │
                    │  ┌─────────────────┐   │
                    │  │   LLM Layers    │   │ ◀── 复用预训练知识
                    │  └────────┬────────┘   │
                    │           ▼            │
                    │  ┌─────────────────┐   │
                    │  │ Speech Decoder  │   │ ◀── XY-Tokenizer
                    │  └─────────────────┘   │
                    └─────────────────────────┘
```

---

## 📊 实施方案对比

### 方案 A：完全替换（激进）

**做法**：关闭 CosyVoice3，完全迁移到 MOSS-Speech

| 优点 | 缺点 |
|------|------|
| 架构简单 | 风险集中 |
| 显存充裕 | 无回退机制 |
| 延迟最优 | 迁移周期长 |

**显存预算**：
- MOSS-Speech 7B FP16: ~15GB
- KV Cache: ~3GB
- 余量: ~6GB ✅

### 方案 B：并行运行（稳健）⭐ 推荐

**做法**：MOSS-Speech 作为新服务，与 CosyVoice3 共存，通过网关路由

```
                         ┌─────────────────┐
                    ┌───▶│  MOSS-Speech    │───▶ 低延迟对话
   用户请求 ──────▶ │    │  (端口 9002)    │
                    │    └─────────────────┘
                    │
              Voice │    ┌─────────────────┐
              Router│───▶│  CosyVoice3     │───▶ 高质量播报
                    │    │  (端口 9001)    │
                    │    └─────────────────┘
                    │
                    │    ┌─────────────────┐
                    └───▶│  IndexTTS/VoxCPM│───▶ 特殊场景
                         │  (端口 9003)    │
                         └─────────────────┘
```

| 优点 | 缺点 |
|------|------|
| 灰度发布 | 显存紧张 |
| 有回退机制 | 架构复杂 |
| 场景分流 | 需要路由逻辑 |

**显存预算**（需要时分复用）：
- MOSS-Speech 7B Int8: ~8GB
- CosyVoice3: ~9GB
- **总计**: ~17GB（需要时分复用，不能同时运行）

### 方案 C：Docker 隔离（工程化）

**做法**：MOSS-Speech 部署在独立容器，通过环境隔离解决依赖冲突

```dockerfile
# MOSS-Speech 专用容器
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel
# ... 独立的依赖环境
```

---

## 📅 分阶段实施计划

### Phase 0：环境验证（1-2天）

**目标**：验证 MOSS-Speech 能否在当前硬件上运行

```bash
# 任务清单
□ 1. 下载 MOSS-Speech 16k 权重
□ 2. 创建隔离的 Python 虚拟环境
□ 3. 测试基础推理（非流式）
□ 4. 测量显存占用和推理延迟
```

**验证命令**：
```bash
# 创建隔离环境
python3 -m venv /workspace/envs/moss_env
source /workspace/envs/moss_env/bin/activate

# 安装依赖
pip install torch==2.1.2+cu121 torchaudio transformers==4.37.0 accelerate einops

# 下载模型（假设已开源）
huggingface-cli download fnlp/MOSS-Speech --local-dir /tmp/MOSS-Speech
```

**成功标准**：
- [ ] 模型加载成功，显存占用 <16GB
- [ ] 单次推理延迟 <500ms
- [ ] 生成的音频可正常播放

---

### Phase 1：流式推理引擎（3-5天）

**目标**：实现流式音频生成，TTFA <300ms

**关键代码模块**：

```
server/cortex/
├── moss_server.py          # 新增：MOSS-Speech 服务入口
├── models/
│   ├── moss_handler.py     # 新增：MOSS-Speech 推理处理器
│   └── xy_tokenizer.py     # 新增：XY-Tokenizer 封装
└── ...
```

**核心实现要点**：

1. **音频预处理**：重采样到 16kHz
```python
# 必须！MOSS-Speech 16k 版本严格要求 16000Hz
resampler = torchaudio.transforms.Resample(source_sr, 16000)
```

2. **流式 Token 生成**：
```python
# 关键参数
generation_kwargs = {
    "max_new_tokens": 2000,
    "do_sample": True,
    "top_p": 0.8,
    "temperature": 0.5,  # 🔴 关键：降低以减少声学幻觉
}
```

3. **分块策略**（文档核心建议）：
```python
# 首块：1-2帧（<160ms）立即发送 - 追求极致 TTFA
# 后续块：3-5帧（240-400ms）- 保证稳定性
FIRST_CHUNK_FRAMES = 2
NORMAL_CHUNK_FRAMES = 4
```

---

### Phase 2：WebSocket 全双工（2-3天）

**目标**：实现真·实时双向通信，支持打断

**协议设计**：

```
Client → Server (上行)：
  - 音频流（16kHz PCM）
  - 控制命令（打断、静音）

Server → Client (下行)：
  - 音频流（16kHz PCM）
  - 状态信息（正在思考、正在说话）
```

**打断机制实现**（Barge-in）：
```python
async def handle_barge_in():
    """当 VAD 检测到用户开始说话时"""
    # 1. 停止当前生成
    model.stop_generate()
    # 2. 清空发送缓冲区
    audio_buffer.clear()
    # 3. 通知客户端
    await websocket.send_json({"event": "interrupted"})
```

---

### Phase 3：集成与路由（2天）

**目标**：与现有系统集成，实现智能路由

**voice_adapter.py 改动**：

```python
class VoiceAdapter:
    def __init__(self):
        self.moss_handler = MOSSHandler()      # 新增
        self.cosy_handler = CosyVoiceHandler() # 保留
        
    async def synthesize(self, text, mode="auto"):
        if mode == "realtime" or (mode == "auto" and self._is_dialogue(text)):
            # 对话场景：使用 MOSS-Speech
            return await self.moss_handler.synthesize_stream(text)
        else:
            # 播报场景：使用 CosyVoice
            return await self.cosy_handler.synthesize(text)
```

---

### Phase 4：优化与调优（持续）

**显存优化**：
```python
# Int8 量化加载（推荐）
model = AutoModelForCausalLM.from_pretrained(
    "fnlp/MOSS-Speech",
    load_in_8bit=True,  # 显存从 15GB 降至 8GB
    device_map="auto",
)
```

**延迟优化检查清单**：
- [x] Flash Attention 2 已启用 ✅ v2.8.3
- [x] KV Cache 预分配 ✅ use_cache=True
- [🔶] 首块激进策略：TTFA=372ms（目标300ms，差72ms）
- [ ] WebSocket 二进制传输（非 Base64）

**实测数据（2026-01-13）**：
| 指标 | 测量值 | 目标值 | 状态 |
|------|--------|--------|------|
| TTFA（首次前向） | 372ms | <300ms | 🔶 差72ms |
| 语音输入 TTFA | 382-388ms | <300ms | 🔶 |
| 端到端延迟 | 23s | - | 需VLLM优化 |
| 输出采样率 | 24kHz | 16kHz | ✅ 超额完成 |
| 显存占用 | 21GB (FP16) | <16GB | ❌ 需Int8 |

**关键发现**：
1. Int8 量化反而更慢（1388ms vs 372ms），因小批量推理开销大
2. torch.compile 导致严重倒退（963秒！），因动态形状反复编译
3. Flash Attention 2 + FP16 是最优配置

---

## 🧪 验收标准

### 功能验收

| 测试项 | 标准 | 方法 |
|--------|------|------|
| 基础对话 | 能正常完成 3 轮对话 | 手动测试 |
| 打断功能 | 用户说话时能立即停止 | 手动测试 |
| 情感感知 | 能感知用户语气变化 | 对比测试 |
| 回退机制 | MOSS 失败时切换到 CosyVoice | 故障注入 |

### 性能验收

| 指标 | 基线 (CosyVoice3) | 目标 (MOSS-Speech) |
|------|-------------------|-------------------|
| TTFA | ~1200ms | **<300ms** |
| RTF | ~1.0 | <0.3 |
| 显存 | ~9GB | <16GB |
| 首包延迟 | ~800ms | **<200ms** |

---

## 📦 资源需求

### 硬件
- RTX 4090 24GB（当前已有 ✅）

### 模型权重
```bash
# 需要下载
fnlp/MOSS-Speech          # ~15GB (FP16)
fnlp/MOSS-Speech-Codec    # ~500MB (XY-Tokenizer)
```

### 依赖环境
```
torch==2.1.2+cu121        # 当前 2.4.1，需评估兼容性
transformers==4.37.0      # 当前版本待确认
flash-attn==2.5.0+        # 当前 2.8.3 ✅
torchaudio==2.1.2
accelerate==0.27.0
einops
websockets
```

---

## 🚦 决策点

### 需要您确认的问题

1. **迁移策略选择**：
   - [ ] 方案 A：完全替换（激进）
   - [ ] 方案 B：并行运行（稳健）⭐ 推荐
   - [ ] 方案 C：Docker 隔离

2. **优先级确认**：
   - 是否接受 16kHz 音质（vs CosyVoice 的 24kHz）？
   - 延迟 <300ms 是否为硬指标？

3. **资源分配**：
   - 是否需要同时保留 CosyVoice3 服务？
   - 如果显存不足，是否接受时分复用？

4. **时间窗口**：
   - Phase 0 验证需要 1-2 天
   - 完整迁移预计 2-3 周

---

## 📚 参考资料

- MOSS-Speech 技术报告（OpenMOSS 团队）
- XY-Tokenizer 编解码文档
- CosyVoice 3.0 对比分析
- RTX 4090 部署最佳实践

---

**文档版本**：v1.0
**编制**：Agent
**审核**：待定

