#!/usr/bin/env python3
"""
CosyVoice 单独测试脚本
快速诊断 CosyVoice 导入和初始化问题
"""

import sys
import os

print("=" * 60)
print("CosyVoice 诊断测试")
print("=" * 60)

# Step 1: 检查 CosyVoice 目录
COSYVOICE_PATH = "/workspace/CosyVoice"
print(f"\n[1] 检查 CosyVoice 目录: {COSYVOICE_PATH}")

if os.path.exists(COSYVOICE_PATH):
    print(f"    ✓ 目录存在")
    # 列出内容
    contents = os.listdir(COSYVOICE_PATH)
    print(f"    内容: {contents[:10]}...")  # 只显示前10个
else:
    print(f"    ✗ 目录不存在！")
    sys.exit(1)

# Step 2: 检查 cosyvoice 模块目录
COSYVOICE_MODULE = os.path.join(COSYVOICE_PATH, "cosyvoice")
print(f"\n[2] 检查 cosyvoice 模块目录: {COSYVOICE_MODULE}")

if os.path.exists(COSYVOICE_MODULE):
    print(f"    ✓ 模块目录存在")
    module_contents = os.listdir(COSYVOICE_MODULE)
    print(f"    内容: {module_contents}")
else:
    print(f"    ✗ 模块目录不存在！")
    print(f"    这说明 CosyVoice 仓库结构不对，或者克隆不完整")
    sys.exit(1)

# Step 3: 添加到 sys.path
print(f"\n[3] 添加 {COSYVOICE_PATH} 到 sys.path")
if COSYVOICE_PATH not in sys.path:
    sys.path.insert(0, COSYVOICE_PATH)
    print(f"    ✓ 已添加")
else:
    print(f"    已在 sys.path 中")

print(f"    当前 sys.path 前5项:")
for i, p in enumerate(sys.path[:5]):
    print(f"      [{i}] {p}")

# Step 4: 尝试导入 cosyvoice
print(f"\n[4] 尝试导入 cosyvoice 模块")

try:
    import cosyvoice
    print(f"    ✓ import cosyvoice 成功")
    print(f"    cosyvoice 位置: {cosyvoice.__file__}")
except ImportError as e:
    print(f"    ✗ import cosyvoice 失败: {e}")

# Step 5: 尝试导入 CosyVoice 类
print(f"\n[5] 尝试导入 CosyVoice 类")

import_attempts = [
    ("cosyvoice.cli.cosyvoice", "CosyVoice"),
    ("cosyvoice.cli.cosyvoice", "CosyVoice2"),
    ("cosyvoice.cosyvoice", "CosyVoice"),
    ("cosyvoice.cosyvoice", "CosyVoice2"),
]

success = False
for module_name, class_name in import_attempts:
    try:
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)
        print(f"    ✓ from {module_name} import {class_name} 成功")
        success = True
        break
    except ImportError as e:
        print(f"    ✗ from {module_name} import {class_name} 失败: {e}")
    except AttributeError as e:
        print(f"    ✗ {module_name} 没有 {class_name}: {e}")

if not success:
    print(f"\n[!] 所有导入方式都失败了")
    print(f"    请检查 CosyVoice 的安装方式和依赖")
    
    # 检查依赖
    print(f"\n[6] 检查关键依赖")
    deps = ["torch", "torchaudio", "onnxruntime", "numpy", "modelscope"]
    for dep in deps:
        try:
            mod = __import__(dep)
            version = getattr(mod, "__version__", "unknown")
            print(f"    ✓ {dep}: {version}")
        except ImportError:
            print(f"    ✗ {dep}: 未安装")
    
    sys.exit(1)

# Step 6: 检查预训练模型
print(f"\n[6] 检查预训练模型")
MODEL_PATH = "/workspace/CosyVoice/pretrained_models/CosyVoice-300M"

if os.path.exists(MODEL_PATH):
    print(f"    ✓ 模型目录存在: {MODEL_PATH}")
    model_contents = os.listdir(MODEL_PATH)
    print(f"    内容: {model_contents[:5]}...")
else:
    print(f"    ✗ 模型目录不存在: {MODEL_PATH}")
    print(f"    需要下载模型")

# Step 7: 尝试初始化模型
print(f"\n[7] 尝试初始化 CosyVoice 模型")
try:
    model = cls(MODEL_PATH)
    print(f"    ✓ 模型初始化成功!")
    print(f"    模型类型: {type(model)}")
except Exception as e:
    print(f"    ✗ 模型初始化失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("诊断完成")
print("=" * 60)



