# 🔮 Project Trinity 运维手册

> **重要**: 本文档记录了所有关键配置、踩坑记录和启动流程。
> 任何 AI Agent 或开发者在操作本项目前，请**务必通读本文档**。

---

## 📐 系统架构

### 微服务架构 (推荐)

```
┌─────────────────────────────────────────────────────────────────┐
│                        RunPod Environment                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────┐  ┌─────────────────────────────┐
│  │   Cortex Model Server       │  │   Trinity Logic Server      │
│  │   (Port 9000)               │  │   (Port 8000)               │
│  │                             │  │                             │
│  │  ┌─────────────────────┐   │  │  ┌─────────────────────┐   │
│  │  │ Brain (Qwen VL)     │   │  │  │ Remote Brain Client │   │
│  │  │ ~14GB VRAM          │   │◄─┼──│ (HTTP → Cortex)     │   │
│  │  └─────────────────────┘   │  │  └─────────────────────┘   │
│  │                             │  │                             │
│  │  ┌─────────────────────┐   │  │  ┌─────────────────────┐   │
│  │  │ Mouth (CosyVoice 3) │   │  │  │ Remote Mouth Client │   │
│  │  │ ~2GB VRAM           │   │◄─┼──│ (HTTP → Cortex)     │   │
│  │  └─────────────────────┘   │  │  └─────────────────────┘   │
│  │                             │  │                             │
│  └─────────────────────────────┘  │  ┌─────────────────────┐   │
│                                    │  │ Voice (SenseVoice)  │   │
│         模型常驻内存                │  │ Local, ~1GB         │   │
│         重启 Logic 不影响           │  └─────────────────────┘   │
│                                    │                             │
│                                    │  ┌─────────────────────┐   │
│                                    │  │ Driver (GeneFace)   │   │
│                                    │  │ Local               │   │
│                                    │  └─────────────────────┘   │
│                                    │                             │
│                                    │  ┌─────────────────────┐   │
│                                    │  │ Mind Engine         │   │
│                                    │  │ BioState, Narrative │   │
│                                    │  └─────────────────────┘   │
│                                    │                             │
│                                    └─────────────────────────────┘
│                                                                  │
│  nginx (系统进程，占用 8001) ← 不要使用此端口！                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 为什么用微服务？

| 场景 | 单体模式 | 微服务模式 |
|------|---------|-----------|
| 修改业务逻辑后重启 | 重新加载所有模型 (~10分钟) | 只重启 Logic Server (~30秒) |
| Pod 休眠唤醒 | 全部重新加载 | Cortex 可单独启动 |
| 调试迭代 | 痛苦 | 快速 |

---

## 🗂️ 关键路径

### 模型存储 (Network Volume)

```
/workspace/models/
├── Qwen2.5-VL-7B-Instruct-AWQ/    # Brain 主模型 (~8GB)
├── CosyVoice3-0.5B/                # Mouth 主模型 (~2GB)
│   ├── cosyvoice3.yaml             # 配置文件
│   ├── config.json                 # Qwen2-0.5B 配置 (重要!)
│   ├── model.safetensors           # Qwen2-0.5B 权重
│   └── CosyVoice-BlankEN/          # 空白音频资源
├── SenseVoiceSmall/                # Voice 模型 (~1GB)
└── LivePortrait_Weights/           # Driver 模型
```

### 代码位置

```
/workspace/project-trinity/project-trinity/
├── server/
│   ├── main.py                     # Logic Server 入口
│   ├── cortex/
│   │   ├── main.py                 # Cortex Model Server 入口
│   │   └── models/
│   │       ├── brain.py            # Qwen VL Handler
│   │       └── mouth.py            # CosyVoice Handler
│   └── adapters/
│       ├── brain_adapter.py        # 支持 remote_url 模式
│       └── mouth_adapter.py        # 支持 remote_url 模式
├── run_microservices.sh            # 一键启动脚本
└── OPERATIONS.md                   # 本文档
```

### Conda 环境

```
/workspace/envs/
├── brain_env/     # 主环境 (Cortex + Logic Server 都用这个)
├── face_env/      # LivePortrait 专用
└── voice_env/     # 备用
```

### CosyVoice 依赖

```
/workspace/CosyVoice/               # CosyVoice 源码 (必须在 sys.path 中)
└── third_party/Matcha-TTS/         # Matcha-TTS 依赖 (也必须在 sys.path 中)
```

---

## 🚀 启动流程

### 方式一：一键启动 (推荐)

```bash
cd /workspace/project-trinity/project-trinity
./scripts/run_microservices.sh
```

### 方式二：手动启动

**Step 1: 启动 Cortex Model Server**
```bash
cd /workspace/project-trinity/project-trinity
export PYTHONPATH="$(pwd)/server:/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS:$PYTHONPATH"
/workspace/envs/brain_env/bin/uvicorn server.cortex.main:app --host 0.0.0.0 --port 9000
```

等待看到：
```
✅ Cortex Server 就绪
INFO:     Uvicorn running on http://0.0.0.0:9000
```

**Step 2: 启动 Trinity Logic Server**
```bash
cd /workspace/project-trinity/project-trinity
export TRINITY_MODE="microservice"
export CORTEX_URL="http://localhost:9000"
export PYTHONPATH="$(pwd)/server:/workspace/CosyVoice:$PYTHONPATH"
/workspace/envs/brain_env/bin/uvicorn server.main:app --host 0.0.0.0 --port 8000
```

等待看到：
```
🎭 Project Trinity 准备就绪!
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 验证服务状态

```bash
# Cortex 健康检查
curl http://localhost:9000/health
# 期望: {"status":"ok","modules":{"brain":true,"mouth":true}}

# Logic Server 健康检查
curl http://localhost:8000/health
# 期望: {"status":"healthy","components":{...all true...}}
```

---

## ⚠️ 踩坑记录 (Critical!)

### 坑1: 端口 8001 被 nginx 占用

**现象**:
```
ERROR: [Errno 98] error while attempting to bind on address ('0.0.0.0', 8001): address already in use
```

**原因**: RunPod 环境的 nginx (PID 47) 占用了 8001 端口。

**解决方案**: Cortex 使用端口 **9000**，不要使用 8001。

**已实现**: `server/cortex/main.py` 中有 Port Guard 机制，启动时会检查并清理端口。

---

### 坑2: CosyVoice 3 必须使用 CosyVoice3 类

**现象**:
```
AssertionError: do not use /workspace/models/CosyVoice3-0.5B for CosyVoice initialization!
```

**原因**: CosyVoice 有三个版本的类：
- `CosyVoice` - 用于 CosyVoice 1.x
- `CosyVoice2` - 用于 CosyVoice 2.x
- `CosyVoice3` - 用于 CosyVoice 3.x (我们用的)

**解决方案**:
```python
from cosyvoice.cli.cosyvoice import CosyVoice3
model = CosyVoice3("/workspace/models/CosyVoice3-0.5B", load_trt=False)
```

---

### 坑3: CosyVoice3 的 Qwen2-0.5B 基座模型

**现象**:
```
huggingface_hub.errors.HFValidationError: Repo id must use alphanumeric chars...
OSError: Error no file named pytorch_model.bin...
```

**原因**: CosyVoice 3 内部使用 Qwen2-0.5B 作为 LLM 组件。需要：
1. `config.json` - 不能为空 `{}`
2. `model.safetensors` - 模型权重

**解决方案**: 确保 `/workspace/models/CosyVoice3-0.5B/` 下有完整的 Qwen2-0.5B 文件：
```bash
# 如果缺失，从 HuggingFace 下载
huggingface-cli download Qwen/Qwen2-0.5B config.json model.safetensors --local-dir /workspace/models/CosyVoice3-0.5B/
```

---

### 坑4: Matcha-TTS 路径必须在 sys.path

**现象**:
```
ModuleNotFoundError: No module named 'matcha.models'
```

**原因**: CosyVoice 3 依赖 Matcha-TTS，但它不在标准 Python 路径中。

**解决方案**: 启动前设置 PYTHONPATH：
```bash
export PYTHONPATH="/workspace/CosyVoice/third_party/Matcha-TTS:$PYTHONPATH"
```

或在代码中：
```python
import sys
sys.path.insert(0, "/workspace/CosyVoice/third_party/Matcha-TTS")
```

---

### 坑5: vLLM GPU 内存不足

**现象**:
```
ValueError: Free memory on device (11.67/23.53 GiB) on startup is less than desired GPU memory utilization (0.6, 14.12 GiB).
```

**原因**: 有残留进程占用 GPU 内存，或 `gpu_memory_utilization` 设置过高。

**解决方案**:
```bash
# 1. 检查 GPU 占用
nvidia-smi

# 2. 杀死所有 Python 进程
pkill -9 -f python
pkill -9 -f uvicorn

# 3. 确认 GPU 清空
nvidia-smi  # 应该显示 0MB / 24576MB

# 4. 重新启动
```

**配置参考** (`config.yaml`):
```yaml
model:
  qwen_gpu_memory_utilization: 0.6  # 24GB GPU 足够
```

---

### 坑6: 文件被意外截断/损坏

**现象**: `IndentationError` 或 `SyntaxError`

**原因**: 编辑操作可能导致文件内容被截断。

**解决方案**:
```bash
# 从 Git 恢复
cd /workspace/project-trinity/project-trinity
git checkout server/main.py

# 或查看 Git diff
git diff server/main.py
```

---

## 🔄 换 Pod 后的检查清单

当你换到新 Pod 时，按顺序执行：

### 1. 检查 Network Volume 挂载
```bash
ls /workspace/models/
# 应该看到: Qwen2.5-VL-7B-Instruct-AWQ, CosyVoice3-0.5B, SenseVoiceSmall 等
```

### 2. 检查 Conda 环境
```bash
ls /workspace/envs/brain_env/bin/python
# 应该存在
```

### 3. 检查 CosyVoice 源码
```bash
ls /workspace/CosyVoice/cosyvoice/cli/cosyvoice.py
# 应该存在
```

### 4. 检查代码仓库
```bash
cd /workspace/project-trinity/project-trinity
git status
# 应该是 clean 或有你的修改
```

### 5. 检查端口占用
```bash
netstat -tlnp | grep -E "8000|8001|9000"
# 8001 可能被 nginx 占用 (正常)
# 8000 和 9000 应该空闲
```

### 6. 启动服务
```bash
./scripts/run_microservices.sh
```

### 7. 验证
```bash
curl http://localhost:9000/health
curl http://localhost:8000/health
```

---

## 📊 环境变量参考

| 变量 | 值 | 说明 |
|------|-----|------|
| `TRINITY_MODE` | `microservice` | 启用微服务模式 |
| `CORTEX_URL` | `http://localhost:9000` | Cortex 服务地址 |
| `PYTHONPATH` | 见下 | Python 模块搜索路径 |

**PYTHONPATH 完整设置**:
```bash
export PYTHONPATH="/workspace/project-trinity/project-trinity/server:/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS:$PYTHONPATH"
```

---

## 🛠️ 常用命令

### 查看日志
```bash
# Cortex 日志
tail -f cortex_startup.log

# Logic Server 日志
tail -f server_startup.log
```

### 重启 Logic Server (不影响模型)
```bash
# 杀掉 Logic Server
pkill -f "uvicorn server.main:app"

# 重新启动
export TRINITY_MODE="microservice"
export CORTEX_URL="http://localhost:9000"
cd /workspace/project-trinity/project-trinity
/workspace/envs/brain_env/bin/uvicorn server.main:app --host 0.0.0.0 --port 8000
```

### 完全重启 (包括模型)
```bash
pkill -9 -f uvicorn
./scripts/run_microservices.sh
```

### 检查 GPU 状态
```bash
nvidia-smi
```

---

## 📝 版本信息

- **Qwen Model**: Qwen2.5-VL-7B-Instruct-AWQ
- **CosyVoice**: 3.0 (CosyVoice3-0.5B)
- **SenseVoice**: SenseVoiceSmall
- **vLLM**: 0.13.0
- **Python**: 3.10 (brain_env)

---

## 🆘 紧急故障排除

如果一切都不工作，执行"核弹重置"：

```bash
# 1. 杀掉所有进程
pkill -9 -f python
pkill -9 -f uvicorn

# 2. 清理 GPU
nvidia-smi  # 确认清空

# 3. 恢复代码
cd /workspace/project-trinity/project-trinity
git checkout .

# 4. 重新启动
./scripts/run_microservices.sh
```

---

**最后更新**: 2026-01-08
**维护者**: Project Trinity Team



