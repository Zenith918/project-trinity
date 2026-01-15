#!/bin/bash
# ============================================
# 👄 Cortex-Mouth Server 启动脚本
# 端口: 9001 | 模型: CosyVoice3-0.5B
# ============================================

cd /workspace/project-trinity/project-trinity

export PYTHONPATH="$(pwd)/server:/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS:$PYTHONPATH"
export PATH="/workspace/bin:$PATH"

echo "👄 启动 Cortex-Mouth (端口 9001)..."
/workspace/envs/brain_env/bin/python -m uvicorn server.cortex.mouth_server:app \
    --host 0.0.0.0 --port 9001








