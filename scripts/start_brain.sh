#!/bin/bash
# ============================================
# 🧠 Cortex-Brain Server 启动脚本
# 端口: 9000 | 模型: Qwen2.5-VL-7B-AWQ
# ============================================

cd /workspace/project-trinity/project-trinity

export PYTHONPATH="$(pwd)/server:$PYTHONPATH"

echo "🧠 启动 Cortex-Brain (端口 9000)..."
/workspace/envs/brain_env/bin/python -m uvicorn server.cortex.brain_server:app \
    --host 0.0.0.0 --port 9000






