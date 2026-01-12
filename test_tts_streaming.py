"""
CosyVoice 3.0 流式推理测试
目标: TTFT (首个音频块) < 200ms
"""

import sys
import time
sys.path.insert(0, "/workspace/CosyVoice")
sys.path.insert(0, "/workspace/CosyVoice/third_party/Matcha-TTS")

import numpy as np

print("=" * 60)
print("CosyVoice 3.0 流式推理测试")
print("=" * 60)

# 1. 加载模型
print("\n[1/3] 加载 CosyVoice 3.0...")
load_start = time.time()

from cosyvoice.cli.cosyvoice import CosyVoice3
model = CosyVoice3("/workspace/models/CosyVoice3-0.5B", load_trt=False)

print(f"✅ 模型加载完成 ({time.time() - load_start:.1f}s)")
print(f"   采样率: {model.sample_rate} Hz")

# 2. 创建静音参考音频
import wave
prompt_wav_path = "/tmp/test_prompt.wav"
sample_rate = 16000
silence = (np.random.randn(sample_rate) * 0.0001).astype(np.float32)
silence_int16 = (silence * 32767).astype(np.int16)
with wave.open(prompt_wav_path, 'w') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sample_rate)
    wf.writeframes(silence_int16.tobytes())
print(f"✅ 参考音频已创建")

# 3. 测试流式推理
test_text = "你好呀，今天心情怎么样？"
instruct_text = "用温柔甜美的女声说"

print(f"\n[2/3] 测试非流式推理 (stream=False)...")
start_time = time.time()
chunks_non_stream = []
for output in model.inference_instruct2(
    tts_text=test_text,
    instruct_text=instruct_text,
    prompt_wav=prompt_wav_path,
    stream=False
):
    if 'tts_speech' in output:
        chunks_non_stream.append(output['tts_speech'].cpu().numpy())
        first_chunk_time = time.time() - start_time
        
total_time_non_stream = time.time() - start_time
print(f"   首个音频块: {first_chunk_time * 1000:.0f}ms")
print(f"   总耗时: {total_time_non_stream:.2f}s")
print(f"   音频块数: {len(chunks_non_stream)}")

print(f"\n[3/3] 测试流式推理 (stream=True)...")
start_time = time.time()
first_chunk_time = None
chunks_stream = []
chunk_times = []

for output in model.inference_instruct2(
    tts_text=test_text,
    instruct_text=instruct_text,
    prompt_wav=prompt_wav_path,
    stream=True  # 关键: 开启流式!
):
    chunk_time = time.time() - start_time
    if 'tts_speech' in output:
        chunks_stream.append(output['tts_speech'].cpu().numpy())
        chunk_times.append(chunk_time)
        
        if first_chunk_time is None:
            first_chunk_time = chunk_time
            print(f"   🎯 首个音频块 (TTFT): {first_chunk_time * 1000:.0f}ms")

total_time_stream = time.time() - start_time

print(f"\n" + "=" * 60)
print("📊 测试结果对比")
print("=" * 60)
print(f"{'指标':<20} {'非流式':<15} {'流式':<15}")
print("-" * 60)
print(f"{'TTFT (首音延迟)':<20} {first_chunk_time*1000 if chunks_non_stream else 'N/A':>10.0f}ms   {chunk_times[0]*1000 if chunk_times else 'N/A':>10.0f}ms")
print(f"{'总耗时':<20} {total_time_non_stream:>10.2f}s    {total_time_stream:>10.2f}s")
print(f"{'音频块数':<20} {len(chunks_non_stream):>10}      {len(chunks_stream):>10}")

if chunk_times:
    print(f"\n📈 流式块时间线:")
    for i, t in enumerate(chunk_times[:5]):
        duration = chunks_stream[i].shape[1] / model.sample_rate * 1000
        print(f"   Chunk {i+1}: {t*1000:>6.0f}ms (音频长度: {duration:.0f}ms)")
    if len(chunk_times) > 5:
        print(f"   ... 共 {len(chunk_times)} 个块")

print(f"\n{'='*60}")
if first_chunk_time and first_chunk_time < 0.2:
    print("✅ 流式方案可行! TTFT < 200ms")
elif first_chunk_time and first_chunk_time < 0.5:
    print("⚠️ TTFT 在 200-500ms，可接受但需优化")
else:
    print("❌ TTFT > 500ms，需要进一步优化")
print("=" * 60)






