from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import os

model_path = "/workspace/models/MOSS-Speech"
quant_path = "/workspace/models/MOSS-Speech-AWQ"
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

print(f"开始量化 {model_path} ...")

# 检查路径
if not os.path.exists(model_path):
    print(f"错误: 模型路径不存在 {model_path}")
    exit(1)

# 加载模型
print("加载模型中...")
try:
    model = AutoAWQForCausalLM.from_pretrained(model_path, **{"low_cpu_mem_usage": True})
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    print("开始量化...")
    model.quantize(tokenizer, quant_config=quant_config)
    
    print(f"保存到 {quant_path} ...")
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)
    
    print("✅ 量化完成！")
except Exception as e:
    print(f"❌ 量化失败: {e}")
    import traceback
    traceback.print_exc()
