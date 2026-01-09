#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 启动 VoxCPM 服务 (端口 9003)
# 依赖: pip install voxcpm
echo "🚀 启动 Cortex-Mouth-Daily (VoxCPM)..."
# 使用 exec 替换 shell 进程，并指定正确的 python 解释器
exec /workspace/envs/brain_env/bin/python -m uvicorn server.cortex.mouth_daily:app --host 0.0.0.0 --port 9003 --workers 1
