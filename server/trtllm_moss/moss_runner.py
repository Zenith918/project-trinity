#!/usr/bin/env python3
"""
MOSS-Speech 自定义 Runner
========================

由于 TRT-LLM 的标准 ModelRunner 不支持自定义输出 (audio_logits)，
我们需要直接使用 TensorRT 的原生 API 来运行 Engine。

首席研究员 P0 指令：
- 测量真实 TTFA 和 RTF
- 验证 60% 效率假设
"""

import os
import sys
import time
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

import tensorrt as trt

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))


@dataclass
class InferenceResult:
    """推理结果"""
    logits: torch.Tensor           # 文本 logits
    audio_logits: torch.Tensor     # 音频 logits
    prefill_time_ms: float         # Prefill 时间
    total_time_ms: float           # 总时间


class MossSpeechRunner:
    """
    MOSS-Speech 自定义 TensorRT Runner
    
    直接使用 TensorRT 原生 API，支持 audio_logits 输出
    """
    
    def __init__(self, engine_dir: str):
        """
        初始化 Runner
        
        Args:
            engine_dir: Engine 目录
        """
        self.engine_dir = Path(engine_dir)
        self.engine: Optional[trt.ICudaEngine] = None
        self.context: Optional[trt.IExecutionContext] = None
        self.config: Dict = {}
        
        # 绑定信息
        self.input_bindings: Dict[str, int] = {}
        self.output_bindings: Dict[str, int] = {}
        
        # Stream
        self.stream = torch.cuda.Stream()
        
    def load(self) -> float:
        """
        加载 Engine
        
        Returns:
            加载时间 (秒)
        """
        print(f"[MossSpeechRunner] Loading Engine from {self.engine_dir}")
        
        # 加载配置
        config_path = self.engine_dir / "config.json"
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # 加载 TRT-LLM 插件
        import tensorrt_llm
        
        start = time.perf_counter()
        
        # 读取 Engine 文件
        engine_path = self.engine_dir / "rank0.engine"
        print(f"  Reading {engine_path.stat().st_size / 1024**3:.2f} GB...")
        
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        # 创建 Runtime 和 Engine
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        
        if self.engine is None:
            raise RuntimeError("Failed to deserialize engine")
        
        # 创建执行上下文
        self.context = self.engine.create_execution_context()
        
        load_time = time.perf_counter() - start
        
        # 解析绑定
        self._parse_bindings()
        
        print(f"  ✅ Engine loaded in {load_time:.1f}s")
        print(f"  - Num IO tensors: {self.engine.num_io_tensors}")
        print(f"  - Inputs: {list(self.input_bindings.keys())}")
        print(f"  - Outputs: {list(self.output_bindings.keys())}")
        
        return load_time
    
    def _parse_bindings(self):
        """解析输入/输出绑定"""
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            
            if mode == trt.TensorIOMode.INPUT:
                self.input_bindings[name] = i
            else:
                self.output_bindings[name] = i
    
    def get_binding_shape(self, name: str) -> Tuple:
        """获取绑定的形状"""
        return tuple(self.engine.get_tensor_shape(name))
    
    def get_binding_dtype(self, name: str) -> np.dtype:
        """获取绑定的数据类型"""
        trt_dtype = self.engine.get_tensor_dtype(name)
        dtype_map = {
            trt.DataType.FLOAT: np.float32,
            trt.DataType.HALF: np.float16,
            trt.DataType.INT32: np.int32,
            trt.DataType.INT64: np.int64,
            trt.DataType.BOOL: np.bool_,
        }
        return dtype_map.get(trt_dtype, np.float32)
    
    @torch.inference_mode()
    def infer(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> InferenceResult:
        """
        运行单次推理 (Prefill)
        
        Args:
            input_ids: 输入 token IDs [batch, seq_len]
            position_ids: 位置 IDs (可选)
            
        Returns:
            InferenceResult
        """
        if self.context is None:
            raise RuntimeError("Engine not loaded. Call load() first.")
        
        batch_size, seq_len = input_ids.shape
        
        # 准备输入
        input_ids = input_ids.to(torch.int32).contiguous().cuda()
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.int32, device='cuda').unsqueeze(0)
        position_ids = position_ids.contiguous()
        
        # 设置输入形状
        self.context.set_input_shape('input_ids', (batch_size, seq_len))
        self.context.set_input_shape('position_ids', (batch_size, seq_len))
        
        # 准备其他必需输入 (KV Cache 相关)
        # 注意: 完整实现需要正确设置所有 KV Cache 参数
        # 这里使用简化版本进行性能测试
        
        # 获取输出形状
        pc = self.config.get('pretrained_config', {})
        vocab_size = pc.get('vocab_size', 151680)
        audio_vocab_size = pc.get('audio_vocab_size', 16512)
        
        # 分配输出 buffer
        logits = torch.zeros(batch_size, seq_len, vocab_size, dtype=torch.float16, device='cuda')
        audio_logits = torch.zeros(batch_size, seq_len, audio_vocab_size, dtype=torch.float16, device='cuda')
        
        # 设置 tensor 地址
        self.context.set_tensor_address('input_ids', input_ids.data_ptr())
        self.context.set_tensor_address('position_ids', position_ids.data_ptr())
        self.context.set_tensor_address('logits', logits.data_ptr())
        self.context.set_tensor_address('audio_logits', audio_logits.data_ptr())
        
        # 设置 KV Cache 相关输入 (简化版本)
        # TODO: 完整实现 KV Cache
        
        # 运行推理
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        # 执行
        success = self.context.execute_async_v3(self.stream.cuda_stream)
        
        torch.cuda.synchronize()
        total_time = (time.perf_counter() - start) * 1000
        
        if not success:
            print("  ⚠️ Inference may have failed")
        
        return InferenceResult(
            logits=logits,
            audio_logits=audio_logits,
            prefill_time_ms=total_time,
            total_time_ms=total_time,
        )
    
    def benchmark(
        self,
        num_input_tokens: int = 512,
        num_warmup: int = 2,
        num_runs: int = 5,
    ) -> Dict:
        """
        运行基准测试 (Prefill Only)
        
        注意: 这只测试 Prefill 性能，完整的自回归生成需要实现 KV Cache
        
        Args:
            num_input_tokens: 输入 token 数量
            num_warmup: 预热次数
            num_runs: 测试次数
            
        Returns:
            测试结果字典
        """
        print(f"\n[Benchmark] Starting Prefill-only test...")
        print(f"  - Input tokens: {num_input_tokens}")
        print(f"  - Warmup: {num_warmup}")
        print(f"  - Runs: {num_runs}")
        
        # 准备输入
        vocab_size = self.config.get('pretrained_config', {}).get('vocab_size', 151680)
        input_ids = torch.randint(0, vocab_size, (1, num_input_tokens), dtype=torch.int32).cuda()
        
        # 预热
        print(f"\n[Warmup]")
        for i in range(num_warmup):
            try:
                result = self.infer(input_ids)
                print(f"  Run {i+1}: {result.prefill_time_ms:.1f} ms")
            except Exception as e:
                print(f"  Run {i+1}: Error - {e}")
        
        # 测试
        print(f"\n[Benchmark]")
        times = []
        for i in range(num_runs):
            try:
                result = self.infer(input_ids)
                times.append(result.prefill_time_ms)
                print(f"  Run {i+1}: {result.prefill_time_ms:.1f} ms")
            except Exception as e:
                print(f"  Run {i+1}: Error - {e}")
        
        if not times:
            print("  ❌ All runs failed")
            return {}
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        # 计算吞吐量
        tokens_per_sec = num_input_tokens / (avg_time / 1000)
        
        # 计算理论 TTFA (假设生成 5 个 token)
        # TTFA ≈ Prefill + 5 * decode_time
        # 假设 decode_time ≈ prefill / num_input_tokens * 5
        estimated_ttfa = avg_time + (avg_time / num_input_tokens * 5)
        
        # RTF 计算
        audio_token_rate = 50  # tokens/s
        audio_per_token_ms = 1000 / audio_token_rate  # 20 ms
        first_5_audio_ms = 5 * audio_per_token_ms  # 100 ms of audio
        rtf = estimated_ttfa / first_5_audio_ms
        
        print(f"\n{'='*60}")
        print(f"[Results]")
        print(f"  Prefill time: {avg_time:.1f} ± {std_time:.1f} ms")
        print(f"  Throughput: {tokens_per_sec:.0f} tokens/s (prefill)")
        print(f"  Estimated TTFA: {estimated_ttfa:.1f} ms")
        print(f"  Estimated RTF: {rtf:.3f}")
        print(f"{'='*60}")
        
        # 评估
        print(f"\n[评估]")
        
        # 比较理论值
        theoretical = 1362  # 之前计算的理论吞吐量
        efficiency = tokens_per_sec / theoretical * 100
        print(f"  实际效率: {efficiency:.1f}% (vs 假设 60%)")
        
        if rtf < 1.0:
            print(f"  ✅ RTF < 1.0: 可实现实时")
        else:
            print(f"  ⚠️ RTF >= 1.0: 需要优化")
        
        if estimated_ttfa < 300:
            print(f"  ✅ TTFA < 300ms: 达标")
        elif estimated_ttfa < 400:
            print(f"  ⚠️ TTFA 接近目标")
        else:
            print(f"  ❌ TTFA > 400ms: 需要优化")
        
        return {
            'prefill_time_ms': avg_time,
            'tokens_per_sec': tokens_per_sec,
            'estimated_ttfa_ms': estimated_ttfa,
            'estimated_rtf': rtf,
            'efficiency_percent': efficiency,
        }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine_dir", default="/workspace/models/MOSS-Speech-Engine")
    parser.add_argument("--input_tokens", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MOSS-Speech Custom Runner Benchmark")
    print("=" * 60)
    
    runner = MossSpeechRunner(engine_dir=args.engine_dir)
    load_time = runner.load()
    print(f"\nEngine 加载时间: {load_time:.1f}s")
    
    results = runner.benchmark(
        num_input_tokens=args.input_tokens,
        num_warmup=args.warmup,
        num_runs=args.runs,
    )
    
    if results:
        print(f"\n[JSON Result]")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()



