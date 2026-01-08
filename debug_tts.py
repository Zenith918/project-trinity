#!/usr/bin/env python3
"""独立 TTS 调试脚本"""
import sys
sys.path.insert(0, "/workspace/CosyVoice")
sys.path.insert(0, "/workspace/CosyVoice/third_party/Matcha-TTS")

import os
import wave
import numpy as np

print("=" * 50)
print("CosyVoice TTS 调试")
print("=" * 50)

# 1. 创建测试音频
print("\n[1] 创建测试音频...")
test_wav = "/tmp/test_prompt.wav"
sample_rate = 16000
silence = (np.random.randn(sample_rate) * 0.001).astype(np.float32)
silence_int16 = (silence * 32767).astype(np.int16)

with wave.open(test_wav, 'w') as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(sample_rate)
    f.writeframes(silence_int16.tobytes())
print(f"✅ 创建: {test_wav}")

# 2. 测试 torchaudio 加载
print("\n[2] 测试 torchaudio...")
import torchaudio
print(f"Version: {torchaudio.__version__}")

# 尝试不同后端
for backend in ['soundfile', 'sox', None]:
    try:
        if backend:
            audio, sr = torchaudio.load(test_wav, backend=backend)
        else:
            audio, sr = torchaudio.load(test_wav)
        print(f"✅ backend={backend}: shape={audio.shape}, sr={sr}")
        break
    except Exception as e:
        print(f"❌ backend={backend}: {str(e)[:80]}")

# 3. 检查 CosyVoice 的 load_wav
print("\n[3] 检查 CosyVoice load_wav...")
try:
    # 先看看源码
    with open("/workspace/CosyVoice/cosyvoice/utils/file_utils.py", "r") as f:
        content = f.read()
    
    # 找 load_wav 函数
    import re
    match = re.search(r'def load_wav.*?(?=\ndef |\nclass |\Z)', content, re.DOTALL)
    if match:
        print("当前 load_wav 实现:")
        lines = match.group(0).split('\n')[:20]
        for line in lines:
            print(f"  {line}")
except Exception as e:
    print(f"❌ 错误: {e}")

print("\n" + "=" * 50)
