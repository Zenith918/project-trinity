#!/bin/bash
# Project Trinity - RunPod 初始化脚本
# 在 RunPod Web Terminal 中运行

echo "🔮 Project Trinity - RunPod 环境初始化"
echo "========================================"

# 1. 克隆项目
echo ""
echo "📦 Step 1: 克隆项目..."
cd /workspace
git clone https://github.com/Zenith918/project-trinity.git
cd project-trinity

# 2. 创建虚拟环境
echo ""
echo "🐍 Step 2: 创建 Python 虚拟环境..."
python3 -m venv venv
source venv/bin/activate

# 3. 升级 pip
echo ""
echo "⬆️ Step 3: 升级 pip..."
pip install --upgrade pip

# 4. 安装基础依赖
echo ""
echo "📚 Step 4: 安装基础依赖..."
pip install fastapi uvicorn websockets pydantic pydantic-settings loguru numpy aiofiles

# 5. 检查 GPU
echo ""
echo "🎮 Step 5: 检查 GPU 状态..."
nvidia-smi

# 6. 安装 PyTorch (CUDA)
echo ""
echo "🔥 Step 6: 安装 PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 7. 测试 CUDA
echo ""
echo "✅ Step 7: 验证 CUDA..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "========================================"
echo "🎉 基础环境初始化完成!"
echo ""
echo "下一步: 安装 AI 模型依赖"
echo "  cd /workspace/project-trinity"
echo "  source venv/bin/activate"
echo "  pip install -r server/requirements.txt"
echo ""
echo "启动服务:"
echo "  cd server"
echo "  python main.py"
echo "========================================"

