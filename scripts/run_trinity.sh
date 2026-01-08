#!/bin/bash
# Project Trinity - Smart Startup Script
# 智能启动脚本：清理 -> 检查 -> 启动

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 端口配置
PORT_TRINITY=8000
PORT_QDRANT_REST=6333
PORT_QDRANT_GRPC=6334

log() {
    echo -e "${GREEN}[Trinity]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[Trinity]${NC} $1"
}

error() {
    echo -e "${RED}[Trinity]${NC} $1"
}

# ==========================================
# 1. 深度清理 (Deep Clean)
# ==========================================
log "🧹 正在执行深度清理..."

# 1.1 杀掉端口占用
kill_port() {
    local port=$1
    local pids=$(lsof -t -i:$port 2>/dev/null)
    if [ -n "$pids" ]; then
        warn "发现端口 $port 被占用 (PID: $pids)，正在强制终止..."
        kill -9 $pids 2>/dev/null || true
    fi
}

kill_port $PORT_TRINITY
kill_port $PORT_QDRANT_REST
kill_port $PORT_QDRANT_GRPC

# 1.2 杀掉相关进程名 (防止僵尸)
pkill -9 -f "uvicorn main:app" 2>/dev/null || true
pkill -9 -f "python -m uvicorn" 2>/dev/null || true
# 小心不要误杀 jupyter
# pkill -f "python" 

# 1.3 清理显存 (最为关键)
# 这一步通常由 kill 进程自动完成，但为了保险，我们等待几秒让驱动回收资源
log "⏳ 等待 GPU 资源释放..."
sleep 3

# 1.4 检查 GPU 状态
gpu_usage=$(nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader | awk '{sum+=$1} END {print sum}')
log "当前 GPU 显存占用: ${gpu_usage} MiB"

if [ "$gpu_usage" -gt 2000 ]; then
    warn "⚠️ 警告: 显存占用仍高达 ${gpu_usage} MiB，可能有顽固进程！"
    warn "尝试查找并显示占用显存的进程："
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
    
    # 询问用户是否继续? (脚本模式下默认继续，但在日志中留痕)
else
    log "✅ GPU 状态良好 (空闲)"
fi

# ==========================================
# 2. 环境检查 (Environment Check)
# ==========================================
log "🛠️ 正在检查运行环境..."

# 检查 Docker (Qdrant)
if ! command -v docker &> /dev/null; then
    warn "Docker 未安装，将跳过 Qdrant 启动 (使用 Mem0 本地模式)"
else
    # 启动 Qdrant
    if ! docker ps | grep -q qdrant; then
        log "启动 Qdrant 容器..."
        docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
            -v /workspace/qdrant_storage:/qdrant/storage \
            qdrant/qdrant:latest 2>/dev/null || docker start qdrant 2>/dev/null || true
    else
        log "Qdrant 已在运行"
    fi
fi

# ==========================================
# 3. 启动服务 (Launch)
# ==========================================
log "🚀 正在启动 Trinity 服务..."

# 设置环境变量
export TRINITY_DEBUG=false
export PYTHONPATH="/workspace/CosyVoice:${PYTHONPATH}"

# 切换目录
cd "$(dirname "$0")/server"

# 使用 nohup 后台启动，并实时输出日志到文件
# 使用 brain_env (包含 vLLM)
/workspace/envs/brain_env/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8000 > ../server_startup.log 2>&1 &

SERVER_PID=$!
log "服务已在后台启动 (PID: $SERVER_PID)"
log "正在追踪日志 (按 Ctrl+C 停止追踪，服务不会停止)..."
echo "---------------------------------------------------"

# 实时追踪日志
tail -f ../server_startup.log



