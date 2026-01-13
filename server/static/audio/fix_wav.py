#!/usr/bin/env python3
"""
修复流式 WAV 文件的头部长度字段
流式接口生成的 WAV 头只包含第一个 chunk 的长度，需要更新为实际总长度
"""
import struct
import sys
import os

def fix_wav_header(input_path, output_path=None):
    """修复 WAV 文件头部的长度字段"""
    if output_path is None:
        output_path = input_path
    
    with open(input_path, 'rb') as f:
        data = f.read()
    
    # 检查是否是有效的 WAV 文件
    if data[:4] != b'RIFF' or data[8:12] != b'WAVE':
        print(f"❌ {input_path}: 不是有效的 WAV 文件")
        return False
    
    # 计算正确的长度
    file_size = len(data)
    riff_size = file_size - 8  # RIFF chunk size = 文件大小 - 8
    
    # 找到 data chunk 的位置
    data_pos = data.find(b'data')
    if data_pos == -1:
        print(f"❌ {input_path}: 找不到 data chunk")
        return False
    
    data_size = file_size - data_pos - 8  # data chunk size = 文件大小 - data位置 - 8
    
    # 修复头部
    fixed_data = bytearray(data)
    # 修复 RIFF chunk size (位置 4-7)
    struct.pack_into('<I', fixed_data, 4, riff_size)
    # 修复 data chunk size (data_pos + 4)
    struct.pack_into('<I', fixed_data, data_pos + 4, data_size)
    
    with open(output_path, 'wb') as f:
        f.write(fixed_data)
    
    # 计算时长
    # 假设 16-bit mono 24kHz
    duration = data_size / (24000 * 2)
    print(f"✅ {os.path.basename(input_path)}: {file_size} bytes, {duration:.2f}s")
    return True

if __name__ == '__main__':
    if len(sys.argv) < 2:
        # 修复当前目录下所有 test_*.wav
        import glob
        files = glob.glob('test_*.wav')
        if not files:
            print("没有找到 test_*.wav 文件")
            sys.exit(1)
        for f in files:
            fix_wav_header(f)
    else:
        for f in sys.argv[1:]:
            fix_wav_header(f)

