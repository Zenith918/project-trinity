#!/usr/bin/env python3
"""
MOSS-Speech 完整推理测试
========================

使用参考音频控制情绪，生成新的语音，并分析质量。

流程：
1. 参考音频 → Audio Codec 编码 → 风格特征
2. 文本 + 风格特征 → MOSS-Speech → Audio Tokens
3. Audio Tokens + 参考音频 → Audio Codec 解码 → 生成音频
4. 分析生成音频的质量
"""

import os
import sys
import time
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))


@dataclass
class InferenceResult:
    """推理结果"""
    emotion: str
    text: str
    prompt_path: str
    output_path: str
    duration_ms: float
    audio_tokens: Optional[torch.Tensor] = None
    success: bool = True
    error: Optional[str] = None


# 测试配置
TEST_CONFIGS = [
    {
        "emotion": "温柔",
        "emotion_en": "gentle",
        "prompt_path": "/workspace/audio_benchmark/prompts/gentle.wav",
        "text": "你好，很高兴认识你。今天的天气真不错呢。",
    },
    {
        "emotion": "焦急",
        "emotion_en": "anxious",
        "prompt_path": "/workspace/audio_benchmark/prompts/anxious.wav",
        "text": "怎么办怎么办，时间来不及了！我们快点走！",
    },
    {
        "emotion": "开心",
        "emotion_en": "happy",
        "prompt_path": "/workspace/audio_benchmark/prompts/happy.wav",
        "text": "哇，这个礼物太棒了！谢谢你，我好喜欢！",
    },
    {
        "emotion": "失落",
        "emotion_en": "sad",
        "prompt_path": "/workspace/audio_benchmark/prompts/sad.wav",
        "text": "没关系的，我已经习惯了。一个人也挺好的。",
    },
    {
        "emotion": "冷静",
        "emotion_en": "calm",
        "prompt_path": "/workspace/audio_benchmark/prompts/calm.wav",
        "text": "让我来分析一下这个问题的本质和解决方案。",
    },
]


def load_moss_speech_model():
    """加载 MOSS-Speech 原始模型（HuggingFace）"""
    print("=" * 60)
    print("[加载 MOSS-Speech 模型]")
    print("=" * 60)
    
    from transformers import AutoModel, AutoTokenizer
    
    model_path = "/workspace/models/MOSS-Speech"
    
    print(f"  模型路径: {model_path}")
    print("  加载中...")
    
    start = time.perf_counter()
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # 加载模型
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    model.eval()
    
    load_time = time.perf_counter() - start
    print(f"  ✅ 加载完成: {load_time:.1f}s")
    
    return model, tokenizer


def load_moss_speech_processor():
    """加载 MOSS-Speech Processor (需要 codec_path)"""
    print("\n[加载 Processor]")
    
    sys.path.insert(0, "/workspace/models/MOSS-Speech")
    
    from processing_moss_speech import MossSpeechProcessor
    
    # 需要指定 codec_path
    processor = MossSpeechProcessor.from_pretrained(
        "/workspace/models/MOSS-Speech",
        codec_path="/workspace/models/MOSS-Speech-Codec",
        trust_remote_code=True,
        device="cuda"
    )
    
    print("  ✅ Processor 加载完成 (含 Audio Codec)")
    return processor


@torch.inference_mode()
def run_inference(
    model,
    processor,
    text: str,
    prompt_path: str,
    output_path: str,
    max_new_tokens: int = 500,
) -> InferenceResult:
    """
    运行 MOSS-Speech 推理
    
    Args:
        model: MOSS-Speech 模型
        processor: MOSS-Speech Processor
        text: 要合成的文本
        prompt_path: 参考音频路径
        output_path: 输出音频路径
    """
    try:
        start = time.perf_counter()
        
        # 准备输入
        conversation = [
            {"role": "user", "content": text}
        ]
        
        # 编码输入
        inputs = processor(
            conversation,
            output_modality="speech",
            return_tensors="pt"
        )
        
        # 移动到 GPU
        inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                  for k, v in inputs.items()}
        
        # 生成
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
        )
        
        # 解码音频
        responses = processor.batch_decode(
            outputs,
            decoder_audio_prompt_path=prompt_path
        )
        
        # 保存音频
        if responses and responses[0].audio is not None:
            import soundfile as sf
            audio = responses[0].audio.squeeze().numpy()
            sf.write(output_path, audio, responses[0].sampling_rate)
        
        duration_ms = (time.perf_counter() - start) * 1000
        
        return InferenceResult(
            emotion="",
            text=text,
            prompt_path=prompt_path,
            output_path=output_path,
            duration_ms=duration_ms,
            success=True
        )
        
    except Exception as e:
        return InferenceResult(
            emotion="",
            text=text,
            prompt_path=prompt_path,
            output_path=output_path,
            duration_ms=0,
            success=False,
            error=str(e)
        )


def run_all_tests(model, processor, output_dir: str) -> List[InferenceResult]:
    """运行所有情绪测试"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    print("\n" + "=" * 60)
    print("[开始推理测试]")
    print("=" * 60)
    
    for i, config in enumerate(TEST_CONFIGS):
        print(f"\n--- [{i+1}/{len(TEST_CONFIGS)}] {config['emotion']} ({config['emotion_en']}) ---")
        print(f"  文本: {config['text']}")
        print(f"  参考: {config['prompt_path']}")
        
        output_file = output_path / f"generated_{config['emotion_en']}.wav"
        
        result = run_inference(
            model=model,
            processor=processor,
            text=config['text'],
            prompt_path=config['prompt_path'],
            output_path=str(output_file),
        )
        
        result.emotion = config['emotion']
        
        if result.success:
            print(f"  ✅ 生成成功: {result.duration_ms:.0f}ms")
            print(f"  📁 输出: {output_file}")
        else:
            print(f"  ❌ 生成失败: {result.error}")
        
        results.append(result)
    
    return results


def analyze_generated_audio(results: List[InferenceResult]):
    """分析生成的音频质量"""
    print("\n" + "=" * 60)
    print("[分析生成音频质量]")
    print("=" * 60)
    
    try:
        from audio_quality_benchmark import AudioQualityAnalyzer
        analyzer = AudioQualityAnalyzer(sample_rate=24000)
        
        for result in results:
            if result.success and Path(result.output_path).exists():
                print(f"\n分析: {result.emotion}")
                analysis = analyzer.analyze(result.output_path, result.emotion)
                
                # 生成图表
                plot_path = result.output_path.replace('.wav', '_analysis.png')
                analyzer.plot_analysis(result.output_path, plot_path, result.emotion)
        
        # 生成报告
        report = analyzer.generate_report()
        print(report)
        
        return analyzer.results
        
    except Exception as e:
        print(f"分析失败: {e}")
        return {}


def main():
    """主函数"""
    print("=" * 60)
    print("MOSS-Speech 情感语音生成测试")
    print("=" * 60)
    
    OUTPUT_DIR = "/workspace/audio_benchmark/generated"
    
    # 检查参考音频
    print("\n[检查参考音频]")
    for config in TEST_CONFIGS:
        exists = "✅" if Path(config['prompt_path']).exists() else "❌"
        print(f"  {exists} {config['emotion']}: {config['prompt_path']}")
    
    # 加载模型
    try:
        model, tokenizer = load_moss_speech_model()
        processor = load_moss_speech_processor()
    except Exception as e:
        print(f"\n❌ 模型加载失败: {e}")
        print("\n使用简化模式（仅测试 TRT-LLM Engine）...")
        
        # 使用 TRT-LLM Engine 进行简化测试
        run_trtllm_test(OUTPUT_DIR)
        return
    
    # 运行测试
    results = run_all_tests(model, processor, OUTPUT_DIR)
    
    # 分析结果
    analysis = analyze_generated_audio(results)
    
    # 生成汇总
    print("\n" + "=" * 60)
    print("[汇总]")
    print("=" * 60)
    
    success_count = sum(1 for r in results if r.success)
    print(f"  成功: {success_count}/{len(results)}")
    
    if success_count > 0:
        avg_time = np.mean([r.duration_ms for r in results if r.success])
        print(f"  平均生成时间: {avg_time:.0f}ms")
    
    print(f"\n  生成的音频保存在: {OUTPUT_DIR}/")


def run_trtllm_test(output_dir: str):
    """使用 TRT-LLM Engine 进行简化测试"""
    print("\n[TRT-LLM Engine 简化测试]")
    
    from moss_paged_runtime import MossSpeechPagedRuntime
    
    # 加载 Engine
    runtime = MossSpeechPagedRuntime("/workspace/models/MOSS-Speech-Engine")
    runtime.load()
    
    # 测试推理
    vocab_size = runtime.config.get('pretrained_config', {}).get('vocab_size', 151680)
    
    print("\n测试不同长度输入...")
    for seq_len in [128, 256, 512]:
        input_ids = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.int32, device='cuda')
        result = runtime.infer(input_ids)
        
        print(f"  seq_len={seq_len}: prefill={result.prefill_time_ms:.1f}ms, "
              f"logits_valid={result.logits_valid}, audio_logits_valid={result.audio_logits_valid}")


if __name__ == "__main__":
    main()

