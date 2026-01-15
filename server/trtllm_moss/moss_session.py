#!/usr/bin/env python3
"""
MOSS-Speech TRT-LLM GenerationSession Runner
=============================================

首席研究员 P0 指令：
- 使用 GenerationSession 替代手动 TensorRT API
- 自动处理 KV Cache 参数
- 实现严谨的输出验证
- 获取真实 TTFA/RTF 数据

目标：
- 消除 "Address is not set" 错误
- 验证 logits.sum() 有具体数值
- 获取 512 tokens Prefill 真实毫秒数
- TTFA < 300ms
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

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

# TRT-LLM imports
import tensorrt_llm
from tensorrt_llm.runtime import (
    GenerationSession,
    SamplingConfig,
    ModelConfig,
)
from tensorrt_llm.builder import Engine


@dataclass
class VerifiedResult:
    """经过验证的推理结果"""
    logits: torch.Tensor
    audio_logits: Optional[torch.Tensor]
    output_ids: torch.Tensor
    
    # 验证结果
    logits_valid: bool
    logits_sum: float
    logits_mean: float
    logits_std: float
    
    # 时间测量
    prefill_time_ms: float
    decode_time_ms: float
    total_time_ms: float
    
    # Token 统计
    input_tokens: int
    output_tokens: int


def verify_output(
    tensor: torch.Tensor,
    name: str,
    expected_shape: Tuple = None,
) -> Tuple[bool, Dict]:
    """
    严谨的输出验证函数
    
    Args:
        tensor: 输出 Tensor
        name: Tensor 名称
        expected_shape: 期望形状 (可选)
        
    Returns:
        (is_valid, stats_dict)
    """
    stats = {
        'name': name,
        'shape': tuple(tensor.shape),
        'dtype': str(tensor.dtype),
        'device': str(tensor.device),
        'sum': 0.0,
        'mean': 0.0,
        'std': 0.0,
        'min': 0.0,
        'max': 0.0,
        'has_nan': False,
        'has_inf': False,
        'is_all_zero': True,
    }
    
    # 检查 NaN 和 Inf
    stats['has_nan'] = bool(torch.isnan(tensor).any())
    stats['has_inf'] = bool(torch.isinf(tensor).any())
    
    # 计算统计量
    tensor_float = tensor.float()
    stats['sum'] = float(tensor_float.sum())
    stats['mean'] = float(tensor_float.mean())
    stats['std'] = float(tensor_float.std())
    stats['min'] = float(tensor_float.min())
    stats['max'] = float(tensor_float.max())
    stats['is_all_zero'] = bool(tensor_float.abs().sum() == 0)
    
    # 验证形状
    shape_valid = True
    if expected_shape is not None:
        shape_valid = tensor.shape == expected_shape
    
    # 综合验证
    is_valid = (
        not stats['has_nan'] and
        not stats['has_inf'] and
        not stats['is_all_zero'] and
        shape_valid
    )
    
    return is_valid, stats


class MossSpeechSession:
    """
    MOSS-Speech GenerationSession 封装
    
    使用 TRT-LLM 标准化接口，自动处理 KV Cache
    """
    
    def __init__(self, engine_dir: str):
        """
        初始化
        
        Args:
            engine_dir: Engine 目录
        """
        self.engine_dir = Path(engine_dir)
        self.session: Optional[GenerationSession] = None
        self.config: Dict = {}
        self.model_config: Optional[ModelConfig] = None
        
        # CUDA Events 用于高精度计时
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        
    def load(self) -> float:
        """
        加载 Engine 并创建 GenerationSession
        
        Returns:
            加载时间 (秒)
        """
        print(f"[MossSpeechSession] Loading from {self.engine_dir}")
        
        # 注册自定义模型 (让 TRT-LLM 识别配置)
        from moss_trtllm_model import register_moss_speech_model
        register_moss_speech_model()
        
        # 加载配置
        config_path = self.engine_dir / "config.json"
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        pc = self.config.get('pretrained_config', {})
        bc = self.config.get('build_config', {})
        
        print(f"  Config loaded:")
        print(f"    - Architecture: {pc.get('architecture')}")
        print(f"    - Hidden size: {pc.get('hidden_size')}")
        print(f"    - Num layers: {pc.get('num_hidden_layers')}")
        print(f"    - Vocab size: {pc.get('vocab_size')}")
        
        start = time.perf_counter()
        
        # 加载 Engine
        engine = Engine.from_dir(str(self.engine_dir))
        
        # 创建 ModelConfig (包含所有必需参数)
        plugin_config = bc.get('plugin_config', {})
        from tensorrt_llm.bindings import KVCacheType
        
        # 确定 KV Cache 类型
        paged_kv = plugin_config.get('paged_kv_cache', True)
        kv_type = KVCacheType.PAGED if paged_kv else KVCacheType.CONTINUOUS
        
        self.model_config = ModelConfig(
            # 必需参数
            max_batch_size=bc.get('max_batch_size', 1),
            max_beam_width=1,
            vocab_size=pc.get('vocab_size', 151680),
            num_layers=pc.get('num_hidden_layers', 40),
            num_heads=pc.get('num_attention_heads', 32),
            num_kv_heads=pc.get('num_key_value_heads', 8),
            hidden_size=pc.get('hidden_size', 4096),
            gpt_attention_plugin=True,  # bool, not string
            # 可选参数
            dtype=pc.get('dtype', 'float16'),
            remove_input_padding=plugin_config.get('remove_input_padding', True),
            kv_cache_type=kv_type,
            cross_attention=False,
            has_position_embedding=False,
            has_token_type_embedding=False,
            max_prompt_embedding_table_size=0,
        )
        
        # 创建 GenerationSession
        # 注意: GenerationSession 需要特定的参数
        try:
            self.session = GenerationSession(
                model_config=self.model_config,
                engine_buffer=engine.engine,  # Engine binary
                mapping=tensorrt_llm.Mapping(
                    world_size=1,
                    rank=0,
                    tp_size=1,
                    pp_size=1,
                ),
            )
            print(f"  ✅ GenerationSession created")
        except Exception as e:
            print(f"  ⚠️ GenerationSession creation failed: {e}")
            print(f"  尝试使用备用方法...")
            self.session = self._create_session_fallback(engine)
        
        load_time = time.perf_counter() - start
        
        print(f"  Total load time: {load_time:.1f}s")
        return load_time
    
    def _create_session_fallback(self, engine: Engine):
        """
        备用方法：直接使用 Engine
        """
        # 由于 GenerationSession 可能不兼容自定义模型，
        # 我们使用 Engine + 手动 KV Cache 管理
        from tensorrt_llm.runtime.kv_cache_manager import KVCacheManager
        
        pc = self.config.get('pretrained_config', {})
        bc = self.config.get('build_config', {})
        
        # 获取配置参数
        num_layers = pc.get('num_hidden_layers', 40)
        num_heads = pc.get('num_attention_heads', 32)
        num_kv_heads = pc.get('num_key_value_heads', 8)
        head_size = pc.get('hidden_size', 4096) // num_heads
        max_batch_size = bc.get('max_batch_size', 1)
        max_seq_len = bc.get('max_seq_len', 4096)
        
        print(f"  Using fallback with KV Cache Manager")
        print(f"    - Layers: {num_layers}")
        print(f"    - KV Heads: {num_kv_heads}")
        print(f"    - Head size: {head_size}")
        
        # 返回一个包含必要信息的对象
        class FallbackSession:
            def __init__(self, eng, cfg):
                self.engine = eng
                self.config = cfg
                self.runtime = None
                self.context = None
                self._setup_context()
            
            def _setup_context(self):
                import tensorrt as trt
                logger = trt.Logger(trt.Logger.WARNING)
                runtime = trt.Runtime(logger)
                self.runtime = runtime
                
                # 反序列化
                if hasattr(self.engine, 'engine'):
                    engine_data = self.engine.engine
                else:
                    engine_path = Path(cfg['engine_dir']) / "rank0.engine"
                    with open(engine_path, 'rb') as f:
                        engine_data = f.read()
                
                self.trt_engine = runtime.deserialize_cuda_engine(engine_data)
                self.context = self.trt_engine.create_execution_context()
        
        return FallbackSession(engine, {'engine_dir': str(self.engine_dir), **self.config})
    
    def setup(
        self,
        batch_size: int = 1,
        max_context_length: int = 2048,
        max_new_tokens: int = 100,
    ):
        """
        设置 Session 参数
        """
        if self.session is None:
            raise RuntimeError("Session not loaded")
        
        if hasattr(self.session, 'setup'):
            self.session.setup(
                batch_size=batch_size,
                max_context_length=max_context_length,
                max_new_tokens=max_new_tokens,
                beam_width=1,
            )
            print(f"  Session setup: batch={batch_size}, ctx={max_context_length}, new_tokens={max_new_tokens}")
    
    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
    ) -> VerifiedResult:
        """
        生成并验证输出
        
        Args:
            input_ids: 输入 token IDs [batch, seq_len]
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_k: Top-K
            top_p: Top-P
            
        Returns:
            VerifiedResult
        """
        if self.session is None:
            raise RuntimeError("Session not loaded")
        
        batch_size, seq_len = input_ids.shape
        input_ids = input_ids.to(torch.int32).cuda()
        
        pc = self.config.get('pretrained_config', {})
        vocab_size = pc.get('vocab_size', 151680)
        audio_vocab_size = pc.get('audio_vocab_size', 16512)
        
        # 配置采样
        sampling_config = SamplingConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        
        # 准备 context_lengths
        context_lengths = torch.tensor([seq_len] * batch_size, dtype=torch.int32, device='cuda')
        
        # 使用 CUDA Events 精确计时
        torch.cuda.synchronize()
        self.start_event.record()
        
        try:
            if hasattr(self.session, 'decode'):
                # 标准 GenerationSession 路径
                outputs = self.session.decode(
                    input_ids=input_ids,
                    context_lengths=context_lengths,
                    sampling_config=sampling_config,
                    output_sequence_lengths=True,
                    return_dict=True,
                )
                
                self.end_event.record()
                torch.cuda.synchronize()
                
                # 解析输出
                output_ids = outputs.get('output_ids', outputs.get('sequences', input_ids))
                logits = outputs.get('logits', torch.zeros(batch_size, seq_len, vocab_size, device='cuda'))
                
            else:
                # Fallback: 使用手动推理
                logits, audio_logits, prefill_time = self._manual_inference(
                    input_ids, context_lengths, vocab_size, audio_vocab_size
                )
                output_ids = input_ids  # 无生成
                
                self.end_event.record()
                torch.cuda.synchronize()
                
        except Exception as e:
            print(f"  ⚠️ Generation error: {e}")
            self.end_event.record()
            torch.cuda.synchronize()
            
            # 返回空结果
            return VerifiedResult(
                logits=torch.zeros(batch_size, seq_len, vocab_size, device='cuda'),
                audio_logits=None,
                output_ids=input_ids,
                logits_valid=False,
                logits_sum=0.0,
                logits_mean=0.0,
                logits_std=0.0,
                prefill_time_ms=0.0,
                decode_time_ms=0.0,
                total_time_ms=0.0,
                input_tokens=seq_len,
                output_tokens=0,
            )
        
        # 计算时间 (毫秒)
        total_time_ms = self.start_event.elapsed_time(self.end_event)
        
        # 验证 logits
        logits_valid, logits_stats = verify_output(
            logits if 'logits' in dir() else torch.zeros(1),
            'logits',
            expected_shape=(batch_size, seq_len, vocab_size) if seq_len > 0 else None,
        )
        
        # 验证 audio_logits (如果存在)
        audio_logits_tensor = None
        if 'audio_logits' in dir() and audio_logits is not None:
            audio_logits_tensor = audio_logits
        
        return VerifiedResult(
            logits=logits if 'logits' in dir() else torch.zeros(batch_size, seq_len, vocab_size, device='cuda'),
            audio_logits=audio_logits_tensor,
            output_ids=output_ids,
            logits_valid=logits_valid,
            logits_sum=logits_stats['sum'],
            logits_mean=logits_stats['mean'],
            logits_std=logits_stats['std'],
            prefill_time_ms=total_time_ms,  # 简化
            decode_time_ms=0.0,
            total_time_ms=total_time_ms,
            input_tokens=seq_len,
            output_tokens=output_ids.shape[-1] - seq_len if output_ids.shape[-1] > seq_len else 0,
        )
    
    def _manual_inference(
        self,
        input_ids: torch.Tensor,
        context_lengths: torch.Tensor,
        vocab_size: int,
        audio_vocab_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        手动推理 (Fallback)
        """
        import tensorrt as trt
        
        batch_size, seq_len = input_ids.shape
        
        # 获取 TRT context
        if hasattr(self.session, 'context'):
            context = self.session.context
            engine = self.session.trt_engine
        else:
            raise RuntimeError("No valid context")
        
        # 准备所有输入
        position_ids = torch.arange(seq_len, dtype=torch.int32, device='cuda').unsqueeze(0)
        last_token_ids = torch.tensor([seq_len - 1], dtype=torch.int32, device='cuda')
        
        # 分配输出
        logits = torch.zeros(batch_size, seq_len, vocab_size, dtype=torch.float16, device='cuda')
        audio_logits = torch.zeros(batch_size, seq_len, audio_vocab_size, dtype=torch.float16, device='cuda')
        
        # 设置所有 tensor 地址
        bindings = {}
        
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = engine.get_tensor_shape(name)
            dtype = engine.get_tensor_dtype(name)
            mode = engine.get_tensor_mode(name)
            
            # 创建或获取 tensor
            if name == 'input_ids':
                tensor = input_ids.contiguous()
            elif name == 'position_ids':
                tensor = position_ids.contiguous()
            elif name == 'last_token_ids':
                tensor = last_token_ids.contiguous()
            elif name == 'context_lengths':
                tensor = context_lengths.contiguous()
            elif name == 'sequence_length':
                tensor = context_lengths.contiguous()
            elif name == 'logits':
                tensor = logits
            elif name == 'audio_logits':
                tensor = audio_logits
            elif 'host_' in name:
                # Host tensors
                if dtype == trt.DataType.INT32:
                    cpu_shape = [d if d > 0 else 1 for d in shape]
                    tensor = torch.zeros(cpu_shape, dtype=torch.int32, device='cpu').contiguous()
                else:
                    cpu_shape = [d if d > 0 else 1 for d in shape]
                    tensor = torch.zeros(cpu_shape, dtype=torch.float32, device='cpu').contiguous()
            else:
                # 其他输入 (KV cache 等)
                if mode == trt.TensorIOMode.INPUT:
                    # 创建空 tensor
                    gpu_shape = [d if d > 0 else 1 for d in shape]
                    if dtype == trt.DataType.INT32:
                        tensor = torch.zeros(gpu_shape, dtype=torch.int32, device='cuda').contiguous()
                    elif dtype == trt.DataType.INT64:
                        tensor = torch.zeros(gpu_shape, dtype=torch.int64, device='cuda').contiguous()
                    else:
                        tensor = torch.zeros(gpu_shape, dtype=torch.float16, device='cuda').contiguous()
                else:
                    continue
            
            # 设置地址
            try:
                if 'host_' in name:
                    context.set_tensor_address(name, tensor.data_ptr())
                else:
                    context.set_tensor_address(name, tensor.data_ptr())
                bindings[name] = tensor
            except Exception as e:
                print(f"    ⚠️ Failed to set {name}: {e}")
        
        # 设置输入形状
        try:
            context.set_input_shape('input_ids', (batch_size, seq_len))
            context.set_input_shape('position_ids', (batch_size, seq_len))
        except:
            pass
        
        # 执行
        stream = torch.cuda.Stream()
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        success = context.execute_async_v3(stream.cuda_stream)
        
        stream.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        
        if not success:
            print("    ⚠️ execute_async_v3 returned False")
        
        return logits, audio_logits, elapsed
    
    def benchmark(
        self,
        num_input_tokens: int = 512,
        max_new_tokens: int = 5,
        num_warmup: int = 3,
        num_runs: int = 10,
    ) -> Dict:
        """
        运行完整基准测试
        
        Args:
            num_input_tokens: 输入 token 数
            max_new_tokens: 生成 token 数 (首批音频 token)
            num_warmup: 预热次数
            num_runs: 测试次数
            
        Returns:
            测试结果字典
        """
        print(f"\n{'='*60}")
        print(f"[Benchmark] MOSS-Speech GenerationSession")
        print(f"{'='*60}")
        print(f"  Input tokens: {num_input_tokens}")
        print(f"  Max new tokens: {max_new_tokens}")
        print(f"  Warmup: {num_warmup}")
        print(f"  Runs: {num_runs}")
        
        # 准备输入
        pc = self.config.get('pretrained_config', {})
        vocab_size = pc.get('vocab_size', 151680)
        
        input_ids = torch.randint(
            0, vocab_size,
            (1, num_input_tokens),
            dtype=torch.int32,
            device='cuda'
        )
        
        # Setup
        self.setup(
            batch_size=1,
            max_context_length=num_input_tokens,
            max_new_tokens=max_new_tokens,
        )
        
        # 预热
        print(f"\n[Warmup]")
        for i in range(num_warmup):
            result = self.generate(input_ids, max_new_tokens=max_new_tokens)
            print(f"  Run {i+1}: {result.total_time_ms:.1f}ms, "
                  f"logits_valid={result.logits_valid}, "
                  f"sum={result.logits_sum:.2e}")
        
        # 正式测试
        print(f"\n[Benchmark Runs]")
        times = []
        valid_count = 0
        sums = []
        
        for i in range(num_runs):
            result = self.generate(input_ids, max_new_tokens=max_new_tokens)
            times.append(result.total_time_ms)
            sums.append(result.logits_sum)
            
            if result.logits_valid:
                valid_count += 1
            
            status = "✓" if result.logits_valid else "✗"
            print(f"  Run {i+1}: {result.total_time_ms:.1f}ms [{status}] "
                  f"sum={result.logits_sum:.2e}, mean={result.logits_mean:.4f}")
        
        # 统计
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        # 计算指标
        tokens_per_sec = num_input_tokens / (avg_time / 1000)
        
        # TTFA 计算 (Prefill + first token)
        ttfa_ms = avg_time  # 简化：假设 prefill 占主导
        
        # RTF 计算
        audio_token_rate = 50  # tokens/s (实时音频)
        audio_duration_ms = max_new_tokens / audio_token_rate * 1000
        rtf = ttfa_ms / audio_duration_ms if audio_duration_ms > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"[Results]")
        print(f"  Time: {avg_time:.1f} ± {std_time:.1f}ms (min={min_time:.1f}, max={max_time:.1f})")
        print(f"  Valid outputs: {valid_count}/{num_runs}")
        print(f"  Throughput: {tokens_per_sec:.0f} tokens/s")
        print(f"  TTFA: {ttfa_ms:.1f}ms")
        print(f"  RTF: {rtf:.3f}")
        print(f"{'='*60}")
        
        # 验证
        print(f"\n[验证]")
        if valid_count == num_runs:
            print(f"  ✅ 所有输出有效 (非零)")
        else:
            print(f"  ⚠️ {num_runs - valid_count}/{num_runs} 输出无效")
        
        if ttfa_ms < 300:
            print(f"  ✅ TTFA < 300ms 达标")
        else:
            print(f"  ⚠️ TTFA > 300ms 未达标")
        
        if rtf < 1.0:
            print(f"  ✅ RTF < 1.0 可实时")
        else:
            print(f"  ⚠️ RTF >= 1.0 需优化")
        
        return {
            'avg_time_ms': avg_time,
            'std_time_ms': std_time,
            'min_time_ms': min_time,
            'max_time_ms': max_time,
            'valid_count': valid_count,
            'total_runs': num_runs,
            'tokens_per_sec': tokens_per_sec,
            'ttfa_ms': ttfa_ms,
            'rtf': rtf,
            'logits_sums': sums,
        }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine_dir", default="/workspace/models/MOSS-Speech-Engine")
    parser.add_argument("--input_tokens", type=int, default=512)
    parser.add_argument("--new_tokens", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MOSS-Speech GenerationSession Benchmark")
    print("=" * 60)
    
    session = MossSpeechSession(engine_dir=args.engine_dir)
    load_time = session.load()
    print(f"\nEngine 加载时间: {load_time:.1f}s")
    
    results = session.benchmark(
        num_input_tokens=args.input_tokens,
        max_new_tokens=args.new_tokens,
        num_warmup=args.warmup,
        num_runs=args.runs,
    )
    
    # 输出 JSON
    print(f"\n[JSON Result]")
    json_result = {k: v for k, v in results.items() if k != 'logits_sums'}
    print(json.dumps(json_result, indent=2))


if __name__ == "__main__":
    main()

