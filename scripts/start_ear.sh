#!/bin/bash
# ============================================
# 👂 Cortex-Ear Server 启动脚本
# 端口: 9002 | 模型: SenseVoiceSmall
# ============================================

cd /workspace/project-trinity/project-trinity

export PYTHONPATH="$(pwd)/server:$PYTHONPATH"

echo "👂 启动 Cortex-Ear (端口 9002)..."
/workspace/envs/brain_env/bin/python -m uvicorn server.cortex.ear_server:app \
    --host 0.0.0.0 --port 9002







