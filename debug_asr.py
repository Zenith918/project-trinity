#!/usr/bin/env python3
"""独立 ASR 调试脚本"""
import sys
import time
import numpy as np

print("=" * 50)
print("SenseVoice ASR 测试")
print("=" * 50)

# 使用刚才 TTS 生成的音频作为测试输入
test_wav = "/tmp/tts_output.wav"

print(f"\n[1] 读取测试音频: {test_wav}")
import wave
with wave.open(test_wav, 'rb') as w:
    sample_rate = w.getframerate()
    frames = w.readframes(w.getnframes())
    audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
print(f"✅ 音频长度: {len(audio_data)/sample_rate:.2f}s, 采样率: {sample_rate}")

print("\n[2] 加载 FunASR/SenseVoice 模型...")
start = time.time()
from funasr import AutoModel

model = AutoModel(
    model="/workspace/models/SenseVoiceSmall",
    trust_remote_code=True,
    device="cuda:0"
)
print(f"✅ 模型加载完成 ({time.time()-start:.1f}s)")

print("\n[3] 测试语音识别...")
start = time.time()

# 需要重采样到 16kHz
if sample_rate != 16000:
    import torchaudio
    import torch
    audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)
    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
    audio_16k = resampler(audio_tensor).squeeze().numpy()
    print(f"   重采样: {sample_rate} -> 16000 Hz")
else:
    audio_16k = audio_data

result = model.generate(
    input=audio_16k,
    cache={},
    language="auto",
    use_itn=True
)

print(f"✅ 识别完成 ({time.time()-start:.2f}s)")
print(f"\n识别结果:")
for r in result:
    text = r.get('text', '')
    print(f"  文本: {text}")

print("\n" + "=" * 50)
print("ASR 测试完成!")
print("=" * 50)
