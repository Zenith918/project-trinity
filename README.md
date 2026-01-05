# 🔮 Project Trinity

**Next-Gen Digital Life Engine | 下一代数字生命引擎**

> 构建拥有生物本能、概率性情绪与长期记忆的有机数字生命

---

## 📖 核心架构：三位一体心智 (The Trinity Mind)

基于 **分层主动推理 (Hierarchical Active Inference)** 理论：

| 层级 | 名称 | 对应组件 | 职责 |
|:-----|:-----|:---------|:-----|
| **Layer 1** | 本我 (The Id) | FunASR (SenseVoice) + Bio-State | 概率内稳态与反射 |
| **Layer 2** | 超我 (The Superego) | Mem0 + Qdrant | 约束与叙事记忆 |
| **Layer 3** | 自我 (The Ego) | Qwen 3 VL + Director Agent | 决策与仲裁 |

---

## 🛠️ 技术栈 (DeepLink 2.5 Stack)

### 服务端 (The Brain - Ubuntu/CUDA)
- **听觉**: FunASR (SenseVoice) - 延迟 <200ms，原生情感识别
- **大脑**: Qwen 3.0-VL (via vLLM) - 视频流理解，高并发
- **嘴巴**: CosyVoice 3.0 (Instruct Mode) - 富情感语音合成
- **神经**: GeneFace++ (Audio2Motion) - 音高感知 FLAME 参数
- **记忆**: Mem0 + Qdrant - 长期记忆图谱

### 客户端 (The Body - Web/Mobile)
- **渲染器**: Three.js + WebGPU 3DGS
- **协议**: WebSocket (Protobuf) - 音频流 + FLAME 参数

---

## 📁 项目结构

```
project-trinity/
├── server/                    # 服务端 (Python/CUDA)
│   ├── adapters/             # AI 模型适配器
│   │   ├── __init__.py
│   │   ├── voice_adapter.py  # FunASR 适配器
│   │   ├── brain_adapter.py  # Qwen VL 适配器
│   │   ├── mouth_adapter.py  # CosyVoice 适配器
│   │   └── driver_adapter.py # GeneFace++ 适配器
│   ├── mind_engine/          # 三位一体心智引擎
│   │   ├── __init__.py
│   │   ├── bio_state.py      # Layer 1: 本我 (概率状态机)
│   │   ├── narrative_mgr.py  # Layer 2: 超我 (记忆管理)
│   │   └── ego_director.py   # Layer 3: 自我 (决策引擎)
│   ├── pipeline/             # 数据流转管线
│   │   ├── __init__.py
│   │   ├── orchestrator.py   # 主编排器
│   │   └── packager.py       # 音视频打包对齐
│   ├── config/               # 配置文件
│   │   └── settings.py
│   ├── main.py               # 服务端入口
│   └── requirements.txt
├── client/                    # 客户端 (Web)
│   ├── src/
│   │   ├── renderer/         # WebGPU 3DGS 渲染
│   │   ├── websocket/        # 通信协议
│   │   └── ui/               # 界面组件
│   ├── public/
│   └── package.json
├── proto/                     # Protobuf 协议定义
│   └── trinity.proto
├── docs/                      # 文档
│   └── architecture.md
├── .gitignore
└── README.md
```

---

## 🚀 快速开始

### 1. 服务端部署 (GPU Server)

```bash
# SSH 连接到 RunPod
ssh root@213.181.111.2 -p 23170

# 克隆仓库
git clone https://github.com/YOUR_USERNAME/project-trinity.git
cd project-trinity/server

# 安装依赖
pip install -r requirements.txt

# 启动服务
python main.py
```

### 2. 客户端运行

```bash
cd client
npm install
npm run dev
```

---

## 📋 开发路线图

- [ ] **Phase 1**: 骨架搭建 - 跑通端云分离链路
- [ ] **Phase 2**: 注入灵魂 - Layer 1 + Layer 3 生物系统
- [ ] **Phase 3**: 记忆进化 - Mem0 长期陪伴

---

## 📜 License

MIT License

