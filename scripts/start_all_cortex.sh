#!/bin/bash
# ============================================
# 🧠👄👂 启动所有 Cortex 服务 (三脑分立)
# ============================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/logs"

mkdir -p "$LOG_DIR"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  🏛️  Trinity Cortex Split Architecture                       ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Brain (9000) | Mouth (9001) | Ear (9002)                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# 检查 GPU
echo "📊 GPU 状态:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
echo ""

# 启动 Brain (最慢，先启动)
echo "🧠 [1/3] 启动 Brain Server..."
nohup bash "$SCRIPT_DIR/start_brain.sh" > "$LOG_DIR/brain.log" 2>&1 &
BRAIN_PID=$!
echo "   PID: $BRAIN_PID"

# 启动 Mouth
echo "👄 [2/3] 启动 Mouth Server..."
nohup bash "$SCRIPT_DIR/start_mouth.sh" > "$LOG_DIR/mouth.log" 2>&1 &
MOUTH_PID=$!
echo "   PID: $MOUTH_PID"

# 启动 Ear
echo "👂 [3/3] 启动 Ear Server..."
nohup bash "$SCRIPT_DIR/start_ear.sh" > "$LOG_DIR/ear.log" 2>&1 &
EAR_PID=$!
echo "   PID: $EAR_PID"

echo ""
echo "⏳ 等待服务就绪..."

# 等待所有服务就绪
for i in {1..60}; do
    brain_ok=$(curl -s http://localhost:9000/health 2>/dev/null | grep -c "ok" || echo "0")
    mouth_ok=$(curl -s http://localhost:9001/health 2>/dev/null | grep -c "ok" || echo "0")
    ear_ok=$(curl -s http://localhost:9002/health 2>/dev/null | grep -c "ok" || echo "0")
    
    status="Brain:${brain_ok} Mouth:${mouth_ok} Ear:${ear_ok}"
    echo -ne "\r   [$i/60] $status"
    
    if [ "$brain_ok" = "1" ] && [ "$mouth_ok" = "1" ] && [ "$ear_ok" = "1" ]; then
        echo ""
        echo ""
        echo "✅ 所有 Cortex 服务就绪!"
        echo ""
        echo "📡 服务端点:"
        echo "   Brain: http://localhost:9000/health"
        echo "   Mouth: http://localhost:9001/health"
        echo "   Ear:   http://localhost:9002/health"
        exit 0
    fi
    sleep 5
done

echo ""
echo "⚠️ 部分服务未就绪，请检查日志:"
echo "   tail -f $LOG_DIR/brain.log"
echo "   tail -f $LOG_DIR/mouth.log"
echo "   tail -f $LOG_DIR/ear.log"







