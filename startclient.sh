#!/bin/bash
# Project Trinity - Phase 1 客户端启动脚本

cd "$(dirname "$0")/client"

echo "🌐 启动客户端服务器 (端口 3000)..."
echo "访问: http://localhost:3000"

python3 -m http.server 3000


