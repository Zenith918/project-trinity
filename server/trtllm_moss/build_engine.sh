#!/bin/bash
# ============================================================
# MOSS-Speech TensorRT-LLM Engine 构建脚本
# ============================================================
# 研究员方案: 将 RTF 4.25 → 0.7
#
# 使用方法:
#   ./build_engine.sh [--fp8] [--int8]
#
# 参数:
#   --fp8   启用 FP8 量化 (推荐, RTX 4090 原生支持)
#   --int8  启用 Weight-Only INT8 量化 (备选)
# ============================================================

set -e

# 配置
MODEL_PATH="/workspace/models/MOSS-Speech"
CHECKPOINT_DIR="/workspace/models/MOSS-Speech-TRTLLM-Checkpoint"
ENGINE_DIR="/workspace/models/MOSS-Speech-TRTLLM-Engine"
DTYPE="float16"
USE_FP8=false
USE_INT8=false

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --fp8)
            USE_FP8=true
            DTYPE="fp8"
            shift
            ;;
        --int8)
            USE_INT8=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "MOSS-Speech TensorRT-LLM Engine 构建"
echo "============================================================"
echo "模型路径: $MODEL_PATH"
echo "数据类型: $DTYPE"
echo "FP8 量化: $USE_FP8"
echo "INT8 量化: $USE_INT8"
echo "============================================================"

# 激活环境
source /workspace/envs/trtllm/bin/activate

# Step 1: 检查模型
echo ""
echo "[Step 1/4] 检查 MOSS-Speech 模型..."
if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "❌ 模型不存在: $MODEL_PATH"
    exit 1
fi
echo "✅ 模型存在"

# Step 2: 转换权重
echo ""
echo "[Step 2/4] 转换权重到 TRT-LLM 格式..."
python3 -c "
from server.trtllm_moss.convert import convert_moss_to_trtllm, ConversionConfig

config = ConversionConfig(
    hf_model_path='$MODEL_PATH',
    output_dir='$CHECKPOINT_DIR',
    dtype='$DTYPE',
    use_fp8=$USE_FP8,
    use_int8_weight_only=$USE_INT8,
)

convert_moss_to_trtllm(config)
"

echo "✅ 权重转换完成: $CHECKPOINT_DIR"

# Step 3: 构建 TensorRT Engine
echo ""
echo "[Step 3/4] 构建 TensorRT Engine..."
echo "⚠️ 这可能需要 10-30 分钟..."

# 使用 trtllm-build CLI (如果可用)
if command -v trtllm-build &> /dev/null; then
    trtllm-build \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --output_dir "$ENGINE_DIR" \
        --max_batch_size 4 \
        --max_input_len 2048 \
        --max_output_len 1024 \
        --paged_kv_cache enable \
        --remove_input_padding enable \
        --gemm_plugin float16 \
        --gpt_attention_plugin float16
else
    echo "⚠️ trtllm-build 不可用，请手动构建:"
    echo "   trtllm-build --checkpoint_dir $CHECKPOINT_DIR --output_dir $ENGINE_DIR"
fi

# Step 4: 验证
echo ""
echo "[Step 4/4] 验证 Engine..."
if [ -f "$ENGINE_DIR/rank0.engine" ]; then
    echo "✅ Engine 构建成功!"
    ls -lh "$ENGINE_DIR/"
else
    echo "⚠️ Engine 文件未找到，可能需要手动构建"
fi

echo ""
echo "============================================================"
echo "构建完成!"
echo ""
echo "下一步: 使用以下代码加载 Engine:"
echo ""
echo "  from tensorrt_llm.runtime import ModelRunner"
echo "  runner = ModelRunner.from_dir('$ENGINE_DIR')"
echo ""
echo "============================================================"



