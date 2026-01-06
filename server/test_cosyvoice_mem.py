import os
import sys
import time
import torch
import subprocess

# 强制添加 CosyVoice 路径
COSYVOICE_PATH = "/workspace/CosyVoice"
sys.path.insert(0, COSYVOICE_PATH)
sys.path.append(os.path.join(COSYVOICE_PATH, "third_party/Matcha-TTS"))

from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav

def check_vram():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
        capture_output=True, text=True
    )
    print(f"🔥 Current VRAM: {result.stdout.strip()} MB")

def test_memory():
    print("=== CosyVoice Memory Profiling ===")
    check_vram()
    
    model_path = "/workspace/CosyVoice/pretrained_models/CosyVoice-300M"
    print(f"Loading CosyVoice from {model_path}...")
    
    # 记录开始时间
    start_time = time.time()
    
    # 初始化
    cosy_voice = CosyVoice(model_path)
    
    print(f"✅ Loaded in {time.time() - start_time:.2f}s")
    check_vram()
    
    print("\nRunning inference...")
    # 简单的推理测试
    prompt_speech_16k = load_wav(os.path.join(COSYVOICE_PATH, 'asset/zero_shot_prompt.wav'), 16000)
    
    for i, j in enumerate(cosy_voice.inference_zero_shot('你好，我是 Trinity，你的数字伴侣。', '希望你今天过得开心。', prompt_speech_16k)):
        print(f"Generated chunk {i}")
        
    print("\n✅ Inference done")
    check_vram()

if __name__ == "__main__":
    test_memory()


