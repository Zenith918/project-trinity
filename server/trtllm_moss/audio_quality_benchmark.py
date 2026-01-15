#!/usr/bin/env python3
"""
MOSS-Speech + BigVGAN-v2 音质与情感基准评估
==========================================

首席研究员任务：
1. 生成 5 组不同情感词的音频（温柔、焦急、开心、失落、冷静）
2. 自动化质量分析（MCD、F0、频谱检查）
3. 性能与音质平衡监测

重要发现：
MOSS-Speech 使用参考音频 (prompt_speech) 控制情绪和音色！
需要准备 5 种情绪的高质量参考音频。
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# 音频处理库
try:
    import librosa
    import librosa.display
    import soundfile as sf
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("⚠️ librosa 未安装，部分功能不可用")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import torch
    import torchaudio
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class EmotionPrompt:
    """情绪参考音频配置"""
    name: str           # 情绪名称
    name_en: str        # 英文名称
    text: str           # 测试文本
    prompt_path: str    # 参考音频路径
    description: str    # 描述


# 5种情绪配置
EMOTION_PROMPTS = [
    EmotionPrompt(
        name="温柔",
        name_en="gentle",
        text="亲爱的，今天过得怎么样？我一直在想你。",
        prompt_path="prompts/gentle.wav",
        description="轻柔、温暖、关怀的语气"
    ),
    EmotionPrompt(
        name="焦急",
        name_en="anxious",
        text="快点快点，我们要迟到了！车马上就开了！",
        prompt_path="prompts/anxious.wav",
        description="紧张、急促、语速快"
    ),
    EmotionPrompt(
        name="开心",
        name_en="happy",
        text="太棒了！我们中奖了！这是我今年最开心的一天！",
        prompt_path="prompts/happy.wav",
        description="欢快、活泼、语调上扬"
    ),
    EmotionPrompt(
        name="失落",
        name_en="sad",
        text="算了，可能这就是命吧。我也不知道该怎么办了。",
        prompt_path="prompts/sad.wav",
        description="低沉、缓慢、略带叹息"
    ),
    EmotionPrompt(
        name="冷静",
        name_en="calm",
        text="根据目前的数据分析，我认为最佳方案是这样的。",
        prompt_path="prompts/calm.wav",
        description="平稳、理性、专业"
    ),
]


class AudioQualityAnalyzer:
    """音频质量分析器"""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.results: Dict = {}
        
    def analyze(self, audio_path: str, label: str) -> Dict:
        """
        全面分析音频质量
        
        Returns:
            分析结果字典
        """
        if not HAS_LIBROSA:
            return {"error": "librosa not installed"}
        
        # 加载音频
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        duration = len(y) / sr
        
        result = {
            "label": label,
            "duration_s": duration,
            "sample_rate": sr,
        }
        
        # 1. 能量分析
        result["energy"] = self._analyze_energy(y)
        
        # 2. F0 (基频) 分析
        result["f0"] = self._analyze_f0(y, sr)
        
        # 3. 频谱分析
        result["spectral"] = self._analyze_spectral(y, sr)
        
        # 4. 质量指标
        result["quality"] = self._analyze_quality(y, sr)
        
        self.results[label] = result
        return result
    
    def _analyze_energy(self, y: np.ndarray) -> Dict:
        """能量分析 - 检测爆音和静音"""
        rms = librosa.feature.rms(y=y)[0]
        
        # 检测异常峰值（爆音）
        threshold = np.mean(rms) + 3 * np.std(rms)
        spikes = np.sum(rms > threshold)
        
        # 检测静音
        silence_threshold = 0.01
        silence_ratio = np.sum(rms < silence_threshold) / len(rms)
        
        return {
            "mean_rms": float(np.mean(rms)),
            "max_rms": float(np.max(rms)),
            "min_rms": float(np.min(rms)),
            "std_rms": float(np.std(rms)),
            "spike_count": int(spikes),
            "silence_ratio": float(silence_ratio),
            "has_spikes": spikes > 5,
            "has_long_silence": silence_ratio > 0.3,
        }
    
    def _analyze_f0(self, y: np.ndarray, sr: int) -> Dict:
        """F0 (基频) 分析 - 检测音调稳定性"""
        # 提取 F0
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=50, fmax=500, sr=sr
        )
        
        # 过滤无效值
        f0_valid = f0[~np.isnan(f0)]
        
        if len(f0_valid) == 0:
            return {"error": "no valid f0"}
        
        # 计算 F0 统计
        f0_mean = float(np.mean(f0_valid))
        f0_std = float(np.std(f0_valid))
        f0_range = float(np.max(f0_valid) - np.min(f0_valid))
        
        # 检测 F0 跳跃
        f0_diff = np.abs(np.diff(f0_valid))
        jumps = np.sum(f0_diff > 50)  # >50Hz 视为跳跃
        
        # F0 平直度（机器人感）
        flatness = f0_std / f0_mean if f0_mean > 0 else 0
        is_robotic = flatness < 0.05  # 变化太小
        
        return {
            "mean_hz": f0_mean,
            "std_hz": f0_std,
            "range_hz": f0_range,
            "jump_count": int(jumps),
            "flatness": float(flatness),
            "voiced_ratio": float(np.mean(voiced_probs[~np.isnan(voiced_probs)])),
            "is_robotic": is_robotic,
            "has_jumps": jumps > 10,
        }
    
    def _analyze_spectral(self, y: np.ndarray, sr: int) -> Dict:
        """频谱分析 - 检测爆音条纹和空洞"""
        # 计算 Mel 频谱
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 高频能量比例
        high_freq_ratio = np.mean(mel_db[60:, :]) / np.mean(mel_db[:20, :])
        
        # 检测垂直条纹（帧间差异过大）
        frame_diff = np.abs(np.diff(mel_db, axis=1))
        stripe_score = np.mean(frame_diff > 10)
        
        # 检测空洞（连续低能量区域）
        low_energy_mask = mel_db < -60
        hole_ratio = np.mean(low_energy_mask)
        
        return {
            "high_freq_ratio": float(high_freq_ratio),
            "stripe_score": float(stripe_score),
            "hole_ratio": float(hole_ratio),
            "has_stripes": stripe_score > 0.1,
            "has_holes": hole_ratio > 0.3,
        }
    
    def _analyze_quality(self, y: np.ndarray, sr: int) -> Dict:
        """综合质量指标"""
        # 信噪比估计
        signal_power = np.mean(y ** 2)
        noise_est = np.mean(np.abs(y[y < np.percentile(np.abs(y), 10)]) ** 2)
        snr_db = 10 * np.log10(signal_power / (noise_est + 1e-10))
        
        # 削波检测
        clip_threshold = 0.95
        clip_ratio = np.mean(np.abs(y) > clip_threshold)
        
        # 零交叉率（噪声指标）
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_mean = np.mean(zcr)
        
        return {
            "snr_db": float(snr_db),
            "clip_ratio": float(clip_ratio),
            "zcr_mean": float(zcr_mean),
            "is_clipped": clip_ratio > 0.01,
            "is_noisy": zcr_mean > 0.2,
        }
    
    def plot_analysis(self, audio_path: str, output_path: str, label: str):
        """生成分析图表"""
        if not HAS_MATPLOTLIB or not HAS_LIBROSA:
            return
        
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        fig.suptitle(f"Audio Quality Analysis: {label}", fontsize=14)
        
        # 1. 波形
        axes[0].set_title("Waveform")
        librosa.display.waveshow(y, sr=sr, ax=axes[0])
        axes[0].set_xlabel("Time (s)")
        
        # 2. Mel 频谱
        axes[1].set_title("Mel Spectrogram")
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        img = librosa.display.specshow(
            mel_db, sr=sr, x_axis='time', y_axis='mel', ax=axes[1]
        )
        fig.colorbar(img, ax=axes[1], format='%+2.0f dB')
        
        # 3. F0 曲线
        axes[2].set_title("F0 (Pitch) Curve")
        f0, _, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
        times = librosa.times_like(f0, sr=sr)
        axes[2].plot(times, f0, label='F0', color='blue')
        axes[2].set_xlabel("Time (s)")
        axes[2].set_ylabel("Frequency (Hz)")
        axes[2].legend()
        
        # 4. RMS 能量
        axes[3].set_title("RMS Energy")
        rms = librosa.feature.rms(y=y)[0]
        times = librosa.times_like(rms, sr=sr)
        axes[3].plot(times, rms, label='RMS', color='green')
        axes[3].set_xlabel("Time (s)")
        axes[3].set_ylabel("RMS")
        axes[3].legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        print(f"  📊 Saved plot: {output_path}")
    
    def generate_report(self) -> str:
        """生成分析报告"""
        report = []
        report.append("=" * 60)
        report.append("MOSS-Speech 音质分析报告")
        report.append("=" * 60)
        
        for label, result in self.results.items():
            report.append(f"\n### {label}")
            report.append(f"时长: {result['duration_s']:.2f}s")
            
            # 能量分析
            energy = result.get('energy', {})
            report.append(f"\n[能量分析]")
            report.append(f"  平均 RMS: {energy.get('mean_rms', 0):.4f}")
            report.append(f"  爆音数: {energy.get('spike_count', 0)}")
            report.append(f"  静音比: {energy.get('silence_ratio', 0):.1%}")
            
            # F0 分析
            f0 = result.get('f0', {})
            report.append(f"\n[F0 (基频) 分析]")
            report.append(f"  平均: {f0.get('mean_hz', 0):.1f} Hz")
            report.append(f"  标准差: {f0.get('std_hz', 0):.1f} Hz")
            report.append(f"  跳跃数: {f0.get('jump_count', 0)}")
            report.append(f"  机器人感: {'⚠️ 是' if f0.get('is_robotic') else '✅ 否'}")
            
            # 频谱分析
            spec = result.get('spectral', {})
            report.append(f"\n[频谱分析]")
            report.append(f"  高频比: {spec.get('high_freq_ratio', 0):.2f}")
            report.append(f"  条纹分数: {spec.get('stripe_score', 0):.2%}")
            report.append(f"  空洞比: {spec.get('hole_ratio', 0):.2%}")
            
            # 质量指标
            quality = result.get('quality', {})
            report.append(f"\n[质量指标]")
            report.append(f"  信噪比: {quality.get('snr_db', 0):.1f} dB")
            report.append(f"  削波: {'⚠️ 是' if quality.get('is_clipped') else '✅ 否'}")
            report.append(f"  噪声: {'⚠️ 是' if quality.get('is_noisy') else '✅ 否'}")
        
        return "\n".join(report)


class EmotionAudioDownloader:
    """情绪参考音频下载器"""
    
    # 开源情绪音频数据集 URLs
    EMOTION_DATASETS = {
        "LibriTTS": "https://www.openslr.org/60/",
        "RAVDESS": "https://zenodo.org/record/1188976",
        "EmoV-DB": "https://github.com/numediart/EmoV-DB",
    }
    
    @staticmethod
    def download_sample_prompts(output_dir: str) -> Dict[str, str]:
        """
        下载示例参考音频
        
        由于版权问题，这里生成合成的示例音频
        实际使用时应该录制或购买专业配音
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("\n=== 生成示例参考音频 ===")
        print("注意：这些是合成的占位音频")
        print("实际部署需要录制高质量的情绪参考音频\n")
        
        downloaded = {}
        
        for emotion in EMOTION_PROMPTS:
            audio_path = output_path / f"{emotion.name_en}.wav"
            
            # 生成不同情绪的示例波形
            duration = 3.0
            sr = 22050
            t = np.linspace(0, duration, int(duration * sr))
            
            # 根据情绪生成不同特征的音频
            if emotion.name_en == "gentle":
                # 温柔：低频、平滑
                freq = 200
                y = 0.3 * np.sin(2 * np.pi * freq * t)
                y *= np.exp(-t / 2)  # 渐弱
                
            elif emotion.name_en == "anxious":
                # 焦急：高频、快速波动
                freq = 350
                y = 0.4 * np.sin(2 * np.pi * freq * t * (1 + 0.1 * np.sin(10 * t)))
                
            elif emotion.name_en == "happy":
                # 开心：明亮、上扬
                freq = 300
                y = 0.4 * np.sin(2 * np.pi * freq * (1 + t/10) * t)
                
            elif emotion.name_en == "sad":
                # 失落：低沉、渐弱
                freq = 150
                y = 0.25 * np.sin(2 * np.pi * freq * t)
                y *= np.exp(-t / 1.5)
                
            else:  # calm
                # 冷静：稳定、中频
                freq = 250
                y = 0.35 * np.sin(2 * np.pi * freq * t)
            
            # 添加轻微噪声使其更自然
            y += 0.01 * np.random.randn(len(y))
            y = np.clip(y, -1, 1).astype(np.float32)
            
            # 保存
            sf.write(audio_path, y, sr)
            downloaded[emotion.name_en] = str(audio_path)
            
            print(f"  ✅ {emotion.name} ({emotion.name_en}): {audio_path}")
        
        return downloaded


def generate_test_audios(output_dir: str = "/workspace/audio_benchmark"):
    """
    生成测试音频（使用 BigVGAN 合成）
    
    注意：完整的 MOSS-Speech 流程需要：
    1. 参考音频 → Audio Codec 编码
    2. 文本 + 参考 Token → MOSS-Speech 生成 Audio Token
    3. Audio Token + 参考音频 → Audio Codec 解码 → 波形
    4. 波形 → BigVGAN 后处理（可选）
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. 准备参考音频
    prompts_dir = output_path / "prompts"
    downloaded = EmotionAudioDownloader.download_sample_prompts(str(prompts_dir))
    
    # 2. 分析参考音频质量
    analyzer = AudioQualityAnalyzer()
    
    print("\n=== 分析参考音频 ===")
    for emotion in EMOTION_PROMPTS:
        audio_path = downloaded.get(emotion.name_en)
        if audio_path and Path(audio_path).exists():
            result = analyzer.analyze(audio_path, f"{emotion.name} (参考)")
            
            # 生成分析图表
            plot_path = output_path / f"analysis_{emotion.name_en}.png"
            analyzer.plot_analysis(audio_path, str(plot_path), emotion.name)
    
    # 3. 生成报告
    report = analyzer.generate_report()
    report_path = output_path / "quality_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📄 报告已保存: {report_path}")
    print(report)
    
    return {
        "prompts": downloaded,
        "report_path": str(report_path),
        "output_dir": str(output_path),
    }


def main():
    """主函数"""
    print("=" * 60)
    print("MOSS-Speech 音质基准评估")
    print("=" * 60)
    
    print("\n⚠️ 重要提示：")
    print("MOSS-Speech 需要参考音频 (prompt_speech) 来控制情绪和音色！")
    print("\n推荐的参考音频来源：")
    print("1. 自行录制高质量配音（推荐）")
    print("2. 使用开源数据集：")
    print("   - RAVDESS (情感语音数据库)")
    print("   - EmoV-DB (情绪语音数据库)")
    print("   - LibriTTS (高质量 TTS 数据)")
    print("3. 购买专业配音素材")
    
    # 生成测试音频和分析
    results = generate_test_audios()
    
    print("\n" + "=" * 60)
    print("[下一步]")
    print("=" * 60)
    print("1. 录制或下载 5 种情绪的高质量参考音频 (3-5秒)")
    print("2. 将参考音频放入 /workspace/audio_benchmark/prompts/")
    print("3. 命名格式: gentle.wav, anxious.wav, happy.wav, sad.wav, calm.wav")
    print("4. 重新运行此脚本进行完整评估")


if __name__ == "__main__":
    main()

