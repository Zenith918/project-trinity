#!/bin/bash
# Project Trinity - Phase 1 服务启动脚本

set -e

# 颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== Project Trinity Phase 1 服务启动 ===${NC}"

# 检查环境变量
export TRINITY_DEBUG=false
export MODEL_qwen_model_path="/workspace/models/Qwen2.5-VL-7B-Instruct-AWQ"
export MODEL_funasr_model="/workspace/models/SenseVoiceSmall"
export MODEL_cosyvoice_model_path="/workspace/models/CosyVoice3-0.5B"
export MODEL_geneface_model_path="/workspace/code/GeneFacePlusPlus"

# 添加 CosyVoice 到 Python 路径
export PYTHONPATH="/workspace/CosyVoice:${PYTHONPATH}"

# 进入 server 目录
cd "$(dirname "$0")/server"

echo -e "${YELLOW}正在启动 Qdrant...${NC}"
# 检查 Qdrant 是否运行
if ! docker ps | grep -q qdrant; then
    echo "启动 Qdrant 向量数据库..."
    docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
        -v /workspace/qdrant_storage:/qdrant/storage qdrant/qdrant:latest 2>/dev/null || \
        docker start qdrant 2>/dev/null || \
        echo -e "${YELLOW}Qdrant 可能已在运行或无法启动 (可在 Debug 模式下继续)${NC}"
else
    echo -e "${GREEN}Qdrant 已在运行${NC}"
fi

echo -e "${YELLOW}正在启动 Trinity 服务...${NC}"

# 使用 brain_env (包含 vLLM) 运行服务
/workspace/envs/brain_env/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8000

echo -e "${GREEN}服务已停止${NC}"


