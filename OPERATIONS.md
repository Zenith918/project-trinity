# 🔮 Project Trinity 运维手册

> **重要**: 本文档记录了所有关键配置、设计原则和踩坑记录。
> 任何 AI Agent 或开发者在操作本项目前，请**务必通读本文档**。

---

## 🔴 核心原则：永远不要频繁重启模型服务

### 为什么？

| 服务 | 启动时间 | 原因 |
|------|----------|------|
| Brain (Qwen2.5-VL) | 3-5 分钟 | vLLM 引擎初始化 + KV Cache 预分配 |
| Mouth-CosyVoice | 2-3 分钟 | 模型加载 + Flow 初始化 |
| Mouth-VoxCPM | 3-5 分钟 | 模型加载 + Warm up |
| Ear (ASR) | 1-2 分钟 | SenseVoice 初始化 |

**每次重启 = 浪费 3-5 分钟 = 研发效率杀手**

### 什么时候才需要重启模型服务？

1. **模型权重更新** - 换了新的 checkpoint
2. **模型代码 Bug 修复** - 推理逻辑本身有错误
3. **依赖库更新** - PyTorch/Transformers 版本变化
4. **显存配置调整** - gpu_memory_utilization 等

**除此之外，一律不要重启！**

---

## 📐 系统架构

### 三脑分立架构 (Trinity Cortex Split)

**核心优势**: 每个模型独立进程，可单独重启，互不干扰。

```
┌─────────────────────────────────────────────────────────────────┐
│                        RunPod Environment                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐    │
│  │ Cortex-Brain    │ │ Cortex-Mouth    │ │ Cortex-Ear      │    │
│  │ (Port 9000)     │ │ (Port 9001/9003)│ │ (Port 9002)     │    │
│  │                 │ │                 │ │                 │    │
│  │ 🧠 Qwen2.5-VL   │ │ 👄 VoxCPM 1.5   │ │ 👂 SenseVoice   │    │
│  │ ~16GB VRAM      │ │ ~2GB VRAM       │ │ ~2GB VRAM       │    │
│  │                 │ │                 │ │                 │    │
│  │ 🔒 常驻运行     │ │ 🔒 常驻运行     │ │ 🔒 常驻运行     │    │
│  └────────┬────────┘ └────────┬────────┘ └────────┬────────┘    │
│           │                   │                   │              │
│           └───────────────────┼───────────────────┘              │
│                               │ HTTP 调用                        │
│                               ▼                                  │
│              ┌─────────────────────────────────┐                 │
│              │   Trinity Logic Server          │                 │
│              │   (Port 8000)                   │                 │
│              │                                 │                 │
│              │  ✅ 改代码后秒级重启             │                 │
│              │  ✅ 所有业务逻辑都在这里         │                 │
│              │  ✅ 不包含任何 ML 模型           │                 │
│              └─────────────────────────────────┘                 │
│                                                                  │
│  ⚠️ nginx 占用 8001，不要使用此端口！                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 研发效率对比

| 场景 | 操作 | 耗时 |
|-----|-----|------|
| 改业务逻辑 | 只重启 Logic Server | ~1秒 |
| 改 TTS 参数 | API 动态传参 | 0秒 |
| 改 LLM Prompt | API 动态传参 | 0秒 |
| 改模型配置 | 重启对应 Cortex | 3-5分钟 |

---

## 🎯 设计规则

### 规则 1: 所有推理参数必须支持 API 动态传递

```python
# ❌ 错误 - 硬编码参数
def synthesize(self, text):
    return self.model.generate(text, steps=12)  # 硬编码！

# ✅ 正确 - 动态参数
def synthesize(self, text, steps=12):
    return self.model.generate(text, inference_timesteps=steps)
```

### 规则 2: 模型服务只做推理，不做业务逻辑

```python
# ❌ 错误 - 业务逻辑在模型服务里
@app.post("/chat")
async def chat(request):
    log_conversation(...)  # 不应该在这里！
    return model.generate(...)

# ✅ 正确 - 纯推理
@app.post("/generate")
async def generate(request):
    return model.generate(text=request["text"], **request.get("params", {}))
```

### 规则 3: 新功能开发流程

```
1. 在 Logic Server 写业务逻辑
2. 用 Mock 数据测试逻辑正确性
3. 连接已运行的模型服务测试
4. 只有模型服务本身有 Bug 才重启它
```

---

## 🗂️ 关键路径

### 模型存储 (Network Volume)

```
/workspace/models/
├── Qwen2.5-VL-7B-Instruct-AWQ/    # Brain (~8GB)
├── VoxCPM1.5/                      # Mouth-Daily (~2GB)
├── CosyVoice3-0.5B/                # Mouth-Backup (~2GB)
├── IndexTTS2.5/                    # Mouth-Emotion (~20GB)
└── SenseVoiceSmall/                # Ear (~1GB)
```

### 代码结构

```
/workspace/project-trinity/project-trinity/
├── server/
│   ├── main.py                     # Logic Server 入口
│   ├── cortex/
│   │   ├── brain_server.py         # Brain 服务 (9000)
│   │   ├── mouth_server.py         # CosyVoice 服务 (9001)
│   │   ├── mouth_daily.py          # VoxCPM 服务 (9003)
│   │   ├── ear_server.py           # ASR 服务 (9002)
│   │   └── models/
│   │       ├── brain.py            # Qwen Handler
│   │       └── mouth.py            # CosyVoice Handler
│   └── adapters/                   # 远程服务适配器
├── scripts/
│   ├── start_voxcpm.sh             # VoxCPM 启动脚本
│   └── setup_runpod.sh             # RunPod 初始化
└── OPERATIONS.md                   # 本文档
```

### 环境

```
/workspace/envs/brain_env/          # 主环境 (所有服务)
/workspace/CosyVoice/               # CosyVoice 源码 (sys.path 依赖)
```

---

## 🚀 启动流程

### Step 1: 启动模型服务 (并行)

```bash
cd /workspace/project-trinity/project-trinity
mkdir -p logs

# 🧠 Brain (端口 9000)
env PYTHONPATH="$(pwd):$(pwd)/server" \
  /workspace/envs/brain_env/bin/python -m uvicorn server.cortex.brain_server:app \
    --host 0.0.0.0 --port 9000 > logs/brain.log 2>&1 &

# 👄 Mouth-VoxCPM (端口 9003) - 推荐
env PYTHONPATH="$(pwd):$(pwd)/server" \
  /workspace/envs/brain_env/bin/python -m uvicorn server.cortex.mouth_daily:app \
    --host 0.0.0.0 --port 9003 > logs/mouth.log 2>&1 &

# 👂 Ear (端口 9002)
env PYTHONPATH="$(pwd):$(pwd)/server" \
  /workspace/envs/brain_env/bin/python -m uvicorn server.cortex.ear_server:app \
    --host 0.0.0.0 --port 9002 > logs/ear.log 2>&1 &
```

### Step 2: 等待就绪

```bash
# 监控
watch -n 5 'for p in 9000 9002 9003; do curl -s http://localhost:$p/health || echo "Port $p: not ready"; done'
```

### Step 3: 启动 Logic Server

```bash
export TRINITY_MODE="microservice"
env PYTHONPATH="$(pwd):$(pwd)/server" \
  /workspace/envs/brain_env/bin/uvicorn server.main:app --host 0.0.0.0 --port 8000
```

---

## 📊 各服务动态参数

### Mouth-VoxCPM (推荐)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| inference_timesteps | 2 | 推理步数 (2=最快, 12=最优质) |
| cfg_value | 1.0 | CFG 引导值 |

**TTFA 测试结果:**
- 2 步: ~450ms ✅
- 4 步: ~560ms
- 8 步: ~780ms
- 12 步: ~1000ms

### Brain (LLM)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| temperature | 0.7 | 生成随机性 |
| max_tokens | 512 | 最大生成长度 |

---

## ⚠️ 踩坑记录

### 坑1: 端口 8001 被 nginx 占用
**解决**: 使用 9000+ 端口

### 坑2: CosyVoice 必须用 CosyVoice3 类
```python
from cosyvoice.cli.cosyvoice import CosyVoice3  # 不是 CosyVoice
```

### 坑3: VoxCPM optimize=True 导致流式失败
```python
# 必须禁用 optimize 才能支持流式
VoxCPM.from_pretrained(..., optimize=False)
```

### 坑4: Matcha-TTS 路径
```bash
export PYTHONPATH="/workspace/CosyVoice/third_party/Matcha-TTS:$PYTHONPATH"
```

### 坑5: 磁盘配额
```bash
# 清理 HF cache
rm -rf ~/.cache/huggingface

# 删除解压后的压缩包
rm -f /workspace/models/*/index-tt2.5.7z
```

---

## 🔄 换 Pod 检查清单

```bash
# 1. 检查模型
ls /workspace/models/

# 2. 检查环境
ls /workspace/envs/brain_env/bin/python

# 3. 检查端口
netstat -tlnp | grep -E "8000|9000"

# 4. 启动服务
# (按上面的启动流程)
```

---

## 🆘 紧急故障排除

```bash
# 核弹重置
pkill -9 -f python
pkill -9 -f uvicorn
nvidia-smi  # 确认 GPU 清空
cd /workspace/project-trinity/project-trinity
git checkout .
# 重新启动
```

---

**最后更新**: 2026-01-09
**维护者**: Project Trinity Team
