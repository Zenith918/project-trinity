#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CosyVoice2 快速测试 - 带详细诊断输出
"""

import os
import sys
import time
import requests
import wave
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════════════════════
CV2_URL = "http://localhost:9005"
PROMPT_WAV = "/workspace/models/CosyVoice/asset/zero_shot_prompt.wav"
PROMPT_TEXT = "希望你以后能够做的比我还好呦。"
# 用更短的文本测试
TEST_TEXT = "你好，这是一个测试。"  # 短文本，快速测试

def log(msg):
    """带时间戳的日志"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()  # 强制刷新，确保立即输出

def main():
    log("=" * 60)
    log("🎤 CosyVoice2 快速测试 (带诊断)")
    log("=" * 60)
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # 1. 检查服务
    # ═══════════════════════════════════════════════════════════════════════════════
    log("[1/4] 检查服务状态...")
    try:
        resp = requests.get(f"{CV2_URL}/health", timeout=5)
        health = resp.json()
        log(f"    ✅ 状态: {health['status']}, 模型: {health['model']}")
    except Exception as e:
        log(f"    ❌ 服务不可达: {e}")
        return
    
    if health['status'] != 'ready':
        log(f"    ❌ 服务未就绪!")
        return
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # 2. 检查参考音频
    # ═══════════════════════════════════════════════════════════════════════════════
    log("[2/4] 检查参考音频...")
    if not os.path.exists(PROMPT_WAV):
        log(f"    ❌ 参考音频不存在: {PROMPT_WAV}")
        return
    
    file_size = os.path.getsize(PROMPT_WAV)
    log(f"    ✅ 参考音频: {PROMPT_WAV}")
    log(f"       大小: {file_size} bytes")
    log(f"       文本: {PROMPT_TEXT}")
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # 3. 非流式测试
    # ═══════════════════════════════════════════════════════════════════════════════
    log("[3/4] 非流式 TTS 测试...")
    log(f"    测试文本: {TEST_TEXT}")
    log(f"    开始请求...")
    
    start_time = time.time()
    try:
        with open(PROMPT_WAV, 'rb') as f:
            files = {'prompt_audio': ('prompt.wav', f, 'audio/wav')}
            data = {
                'text': TEST_TEXT,
                'prompt_text': PROMPT_TEXT,
                'speed': 1.0
            }
            log(f"    发送 POST 请求到 {CV2_URL}/tts ...")
            resp = requests.post(f"{CV2_URL}/tts", files=files, data=data, timeout=300)
        
        elapsed = time.time() - start_time
        log(f"    收到响应，耗时: {elapsed*1000:.0f}ms, 状态: {resp.status_code}")
        
        if resp.status_code != 200:
            log(f"    ❌ 请求失败: {resp.text[:200]}")
            return
        
        # 保存音频
        output_file = "/tmp/cv2_quick_output.wav"
        with open(output_file, 'wb') as f:
            f.write(resp.content)
        
        # 分析音频
        try:
            with wave.open(output_file, 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                audio_duration = frames / rate
        except Exception as e:
            log(f"    ⚠️ 无法解析 WAV: {e}")
            audio_duration = 0
        
        # 计算指标
        rtf = elapsed / audio_duration if audio_duration > 0 else 0
        
        log("")
        log("📊 非流式结果:")
        log(f"    总耗时:   {elapsed*1000:.0f} ms")
        log(f"    音频时长: {audio_duration:.2f} s")
        log(f"    RTF:      {rtf:.3f}")
        log(f"    音频大小: {len(resp.content) / 1024:.1f} KB")
        log(f"    输出文件: {output_file}")
        
        if rtf < 1.0:
            log(f"    ✅ RTF < 1.0，可实时播放")
        else:
            log(f"    ❌ RTF >= 1.0，无法实时！")
        
    except requests.exceptions.Timeout:
        elapsed = time.time() - start_time
        log(f"    ❌ 请求超时! 已等待 {elapsed:.0f}s")
        return
    except Exception as e:
        log(f"    ❌ 请求异常: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # 4. 流式测试
    # ═══════════════════════════════════════════════════════════════════════════════
    log("")
    log("[4/4] 流式 TTS 测试...")
    log(f"    测试文本: {TEST_TEXT}")
    log(f"    开始请求...")
    
    start_time = time.time()
    first_chunk_time = None
    total_bytes = 0
    chunk_count = 0
    pcm_data = bytearray()
    
    try:
        with open(PROMPT_WAV, 'rb') as f:
            files = {'prompt_audio': ('prompt.wav', f, 'audio/wav')}
            data = {
                'text': TEST_TEXT,
                'prompt_text': PROMPT_TEXT,
                'speed': 1.0
            }
            log(f"    发送 POST 请求到 {CV2_URL}/tts/stream ...")
            resp = requests.post(f"{CV2_URL}/tts/stream", files=files, data=data, stream=True, timeout=300)
            
            log(f"    响应状态: {resp.status_code}")
            if resp.status_code != 200:
                log(f"    ❌ 请求失败: {resp.text[:200]}")
                return
            
            sample_rate = int(resp.headers.get('X-Sample-Rate', 44100))
            log(f"    采样率: {sample_rate}Hz")
            log(f"    开始接收数据块...")
            
            for chunk in resp.iter_content(chunk_size=4096):
                if chunk:
                    chunk_count += 1
                    total_bytes += len(chunk)
                    pcm_data.extend(chunk)
                    
                    if first_chunk_time is None:
                        first_chunk_time = time.time() - start_time
                        log(f"    ⚡ 首包到达! TTFA = {first_chunk_time*1000:.0f} ms")
                    
                    # 每收到 10 个块打印一次进度
                    if chunk_count % 10 == 0:
                        elapsed = time.time() - start_time
                        log(f"    💓 已收到 {chunk_count} 块, {total_bytes/1024:.1f}KB, 耗时 {elapsed:.1f}s")
        
        stream_total_time = time.time() - start_time
        
        # 计算音频时长
        stream_audio_duration = len(pcm_data) / (sample_rate * 2)
        stream_rtf = stream_total_time / stream_audio_duration if stream_audio_duration > 0 else 0
        
        # 保存为 WAV
        stream_output = "/tmp/cv2_quick_stream_output.wav"
        with wave.open(stream_output, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(bytes(pcm_data))
        
        log("")
        log("📊 流式结果:")
        log(f"    TTFA:     {first_chunk_time*1000:.0f} ms ⭐")
        log(f"    总耗时:   {stream_total_time*1000:.0f} ms")
        log(f"    音频时长: {stream_audio_duration:.2f} s")
        log(f"    RTF:      {stream_rtf:.3f}")
        log(f"    数据块数: {chunk_count}")
        log(f"    数据大小: {total_bytes/1024:.1f} KB")
        log(f"    输出文件: {stream_output}")
        
        if first_chunk_time and first_chunk_time < 0.5:
            log(f"    ✅ TTFA < 500ms，延迟良好")
        else:
            log(f"    ⚠️ TTFA >= 500ms，延迟较高")
        
    except requests.exceptions.Timeout:
        elapsed = time.time() - start_time
        log(f"    ❌ 请求超时! 已等待 {elapsed:.0f}s, 已收到 {chunk_count} 块, {total_bytes/1024:.1f}KB")
    except Exception as e:
        log(f"    ❌ 请求异常: {e}")
        import traceback
        traceback.print_exc()
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # 完成
    # ═══════════════════════════════════════════════════════════════════════════════
    log("")
    log("=" * 60)
    log("✅ 测试完成!")
    log("=" * 60)
    log("")
    log("🎵 生成的音频文件:")
    log(f"    参考音频:   {PROMPT_WAV}")
    log(f"    非流式输出: /tmp/cv2_quick_output.wav")
    log(f"    流式输出:   /tmp/cv2_quick_stream_output.wav")

if __name__ == "__main__":
    main()



