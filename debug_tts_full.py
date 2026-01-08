#!/usr/bin/env python3
"""完整 CosyVoice TTS 测试"""
import sys
sys.path.insert(0, "/workspace/CosyVoice")
sys.path.insert(0, "/workspace/CosyVoice/third_party/Matcha-TTS")

import os
import time
import wave
import numpy as np

print("=" * 50)
print("CosyVoice 3.0 完整测试")
print("=" * 50)

# 创建静音参考音频
prompt_wav = "/tmp/trinity_prompt.wav"
sample_rate = 16000
silence = (np.random.randn(sample_rate) * 0.001).astype(np.float32)
silence_int16 = (silence * 32767).astype(np.int16)
with wave.open(prompt_wav, 'w') as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(sample_rate)
    f.writeframes(silence_int16.tobytes())
print(f"✅ 参考音频: {prompt_wav}")

# 加载 CosyVoice3
print("\n[1] 加载 CosyVoice3 模型...")
start = time.time()
from cosyvoice.cli.cosyvoice import CosyVoice3

model = CosyVoice3("/workspace/models/CosyVoice3-0.5B", load_trt=False)
print(f"✅ 模型加载完成 ({time.time()-start:.1f}s)")
print(f"   采样率: {model.sample_rate}")

# 测试推理
print("\n[2] 测试语音合成...")
text = "你好，我是Trinity，很高兴认识你！"
instruct = "用温柔甜美的女声说"

start = time.time()
full_audio = []
for output in model.inference_instruct2(
    tts_text=text,
    instruct_text=instruct,
    prompt_wav=prompt_wav,
    stream=False
):
    if 'tts_speech' in output:
        full_audio.append(output['tts_speech'].cpu().numpy())

if full_audio:
    audio_data = np.concatenate(full_audio, axis=1)
    duration = audio_data.shape[1] / model.sample_rate
    print(f"✅ 合成成功!")
    print(f"   耗时: {time.time()-start:.2f}s")
    print(f"   音频时长: {duration:.2f}s")
    print(f"   音频形状: {audio_data.shape}")
    
    # 保存为 WAV
    output_wav = "/tmp/tts_output.wav"
    audio_int16 = (audio_data.flatten() * 32767).astype(np.int16)
    with wave.open(output_wav, 'w') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(model.sample_rate)
        f.writeframes(audio_int16.tobytes())
    print(f"   保存到: {output_wav}")
else:
    print("❌ 合成失败: 无音频输出")

print("\n" + "=" * 50)
print("测试完成!")
print("=" * 50)
