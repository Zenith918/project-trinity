#!/bin/bash
# ============================================================
# Project Trinity - Microservices Startup Script
# ============================================================
# 架构：Cortex (Model Server) + Trinity (Logic Server)
#
# 使用方法:
#   cd /workspace/project-trinity/project-trinity
#   ./run_microservices.sh
#
# 端口分配:
#   - Cortex (模型服务): 9000 (不要用8001，被nginx占用)
#   - Logic  (逻辑服务): 8000
#
# 详细文档请参考: OPERATIONS.md
# ============================================================

set -e  # 遇到错误立即退出

# 切换到项目根目录
cd "$(dirname "$0")"
PROJECT_ROOT=$(pwd)

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# 端口配置
PORT_LOGIC=8000
PORT_CORTEX=9000  # Changed from 8001 (nginx conflict) to 9000
PORT_QDRANT_REST=6333

log() { echo -e "${GREEN}[Trinity]${NC} $1"; }
warn() { echo -e "${YELLOW}[Trinity]${NC} $1"; }
error() { echo -e "${RED}[Trinity]${NC} $1"; }

# ==========================================
# 0. 前置检查 (Pre-flight Checks)
# ==========================================
log "🔍 执行前置检查..."

# 检查模型目录
if [ ! -d "/workspace/models/Qwen2.5-VL-7B-Instruct-AWQ" ]; then
    error "❌ 未找到 Qwen 模型目录: /workspace/models/Qwen2.5-VL-7B-Instruct-AWQ"
    exit 1
fi

if [ ! -d "/workspace/models/CosyVoice3-0.5B" ]; then
    error "❌ 未找到 CosyVoice 模型目录: /workspace/models/CosyVoice3-0.5B"
    exit 1
fi

# 检查 CosyVoice 源码
if [ ! -f "/workspace/CosyVoice/cosyvoice/cli/cosyvoice.py" ]; then
    error "❌ 未找到 CosyVoice 源码: /workspace/CosyVoice"
    exit 1
fi

# 检查 Conda 环境
if [ ! -f "/workspace/envs/brain_env/bin/python" ]; then
    error "❌ 未找到 brain_env 环境: /workspace/envs/brain_env"
    exit 1
fi

log "✅ 前置检查通过"

# ==========================================
# 1. 清理与环境检查
# ==========================================
log "🧹 执行深度清理..."

# 杀端口
fuser -k -n tcp $PORT_LOGIC 2>/dev/null || true
fuser -k -n tcp $PORT_CORTEX 2>/dev/null || true

# 杀进程
pkill -9 -f "uvicorn main:app" 2>/dev/null || true
pkill -9 -f "cortex.main:app" 2>/dev/null || true

# 清理显存
log "⏳ 等待 GPU 资源释放..."
sleep 3
gpu_usage=$(nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader | awk '{sum+=$1} END {print sum}')
log "当前显存占用: ${gpu_usage} MiB"

if [ "$gpu_usage" -gt 2000 ]; then
    warn "⚠️ 显存未完全释放，强制清理中..."
    fuser -k -9 /dev/nvidia0 2>/dev/null || true
fi

# Qdrant 检查
if ! docker ps | grep -q qdrant; then
    log "启动 Qdrant..."
    docker start qdrant 2>/dev/null || true
fi

# ==========================================
# 2. 启动 Cortex (Model Server) - 9000
# ==========================================
log "🧠 启动 Cortex Model Server (Port $PORT_CORTEX)..."
log "   包含: Qwen2.5-VL (Brain) + CosyVoice 3 (Mouth)"

# 设置完整的 PYTHONPATH (包含 Matcha-TTS 依赖)
export PYTHONPATH="$PROJECT_ROOT/server:/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS:$PYTHONPATH"
export TRINITY_DEBUG=false

# 后台启动 Cortex
nohup /workspace/envs/brain_env/bin/uvicorn server.cortex.main:app --host 0.0.0.0 --port $PORT_CORTEX > cortex_startup.log 2>&1 &
CORTEX_PID=$!

log "⏳ 等待 Cortex 就绪 (PID: $CORTEX_PID)..."

# 健康检查循环
MAX_RETRIES=60 # 5分钟
COUNT=0
CORTEX_READY=false

while [ $COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:$PORT_CORTEX/health | grep -q "ok"; then
        CORTEX_READY=true
        break
    fi
    echo -n "."
    sleep 5
    COUNT=$((COUNT+1))
done
echo ""

if [ "$CORTEX_READY" = true ]; then
    log "✅ Cortex Model Server 已就绪!"
else
    error "❌ Cortex 启动超时，请检查 cortex_startup.log"
    tail -n 20 cortex_startup.log
    exit 1
fi

# ==========================================
# 3. 启动 Trinity (Logic Server) - 8000
# ==========================================
log "🚀 启动 Trinity Logic Server (Port $PORT_LOGIC)..."

# 设置环境变量告诉 Logic Server 使用远程模型
export TRINITY_MODE="microservice"
export CORTEX_URL="http://localhost:$PORT_CORTEX"

cd "$PROJECT_ROOT/server"
nohup /workspace/envs/brain_env/bin/uvicorn main:app --host 0.0.0.0 --port $PORT_LOGIC > "$PROJECT_ROOT/server_startup.log" 2>&1 &
LOGIC_PID=$!

# 等待 Logic Server 健康
sleep 10
LOGIC_READY=false
for i in {1..12}; do
    if curl -s http://localhost:$PORT_LOGIC/health | grep -q "healthy"; then
        LOGIC_READY=true
        break
    fi
    echo -n "."
    sleep 5
done
echo ""

if [ "$LOGIC_READY" = true ]; then
    log "✅ Trinity Logic Server 已就绪!"
else
    warn "⚠️ Logic Server 可能仍在初始化中..."
fi

log "=============================================="
log "🎉 Project Trinity 微服务架构启动完成!"
log "=============================================="
log ""
log "服务状态:"
log "   - Cortex Model Server: http://localhost:$PORT_CORTEX (PID: $CORTEX_PID)"
log "   - Trinity Logic Server: http://localhost:$PORT_LOGIC (PID: $LOGIC_PID)"
log ""
log "健康检查:"
log "   curl http://localhost:$PORT_CORTEX/health"
log "   curl http://localhost:$PORT_LOGIC/health"
log ""
log "日志文件:"
log "   - Cortex: cortex_startup.log"
log "   - Logic:  server_startup.log"
log ""
log "提示: 如需修改业务逻辑，只需重启 Logic Server:"
log "   pkill -f 'uvicorn main:app' && cd server && uvicorn main:app --port 8000"
log "=============================================="
log ""
log "正在追踪 Logic Server 日志 (Ctrl+C 退出)..."
echo "---------------------------------------------------"

tail -f "$PROJECT_ROOT/server_startup.log"

