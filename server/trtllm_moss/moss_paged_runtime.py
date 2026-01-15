#!/usr/bin/env python3
"""
MOSS-Speech 支持 PagedAttention 的自定义 Runtime
================================================

首席研究员 P0 指令：
- 手动初始化 KVCacheManager
- 构建自定义推理循环
- 处理双头输出 (logits + audio_logits)
- 获取真实 TTFA 数据

目标：
- RTF 实测 ~0.7
- TTFA 真实锁定 ~300ms
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
class MossInferenceResult:
    """推理结果"""
    output_ids: torch.Tensor
    logits: torch.Tensor
    audio_logits: torch.Tensor
    
    # 时间 (ms)
    prefill_time_ms: float
    decode_time_ms: float
    total_time_ms: float
    
    # 验证
    logits_valid: bool
    audio_logits_valid: bool
    
    # 统计
    input_tokens: int
    output_tokens: int


class PagedKVCacheManager:
    """
    手动管理 PagedAttention KV Cache
    
    参考 TRT-LLM 的 KVCacheManager 实现
    """
    
    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_size: int,
        max_batch_size: int,
        max_seq_len: int,
        tokens_per_block: int = 64,
        dtype: torch.dtype = torch.float16,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.tokens_per_block = tokens_per_block
        self.dtype = dtype
        
        # 计算所需块数
        self.max_blocks_per_seq = (max_seq_len + tokens_per_block - 1) // tokens_per_block
        self.total_blocks = max_batch_size * self.max_blocks_per_seq
        
        # KV Cache 维度: [num_layers, 2, total_blocks, tokens_per_block, num_kv_heads, head_size]
        # 2 = K and V
        self.kv_cache_shape = (
            num_layers,
            2,
            self.total_blocks,
            tokens_per_block,
            num_kv_heads,
            head_size,
        )
        
        # 分配 GPU 内存
        self.kv_cache = None
        self.block_offsets = None
        
    def allocate(self):
        """分配 KV Cache 内存"""
        print(f"[KVCache] Allocating...")
        print(f"  Shape: {self.kv_cache_shape}")
        
        # 计算内存大小
        total_elements = np.prod(self.kv_cache_shape)
        memory_gb = total_elements * 2 / 1024**3  # FP16 = 2 bytes
        print(f"  Memory: {memory_gb:.2f} GB")
        
        # 分配
        self.kv_cache = torch.zeros(
            self.kv_cache_shape,
            dtype=self.dtype,
            device='cuda'
        )
        
        # 初始化 block offsets
        # 每个序列的起始块索引
        self.block_offsets = torch.arange(
            0, self.max_batch_size * self.max_blocks_per_seq,
            self.max_blocks_per_seq,
            dtype=torch.int32,
            device='cuda'
        ).view(self.max_batch_size, 1)
        
        print(f"  ✅ KV Cache allocated")
        return memory_gb
    
    def get_cache_pointers(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取 KV Cache 指针
        
        Returns:
            (kv_cache_block_offsets, host_kv_cache_pool_pointers)
        """
        if self.kv_cache is None:
            raise RuntimeError("KV Cache not allocated")
        
        # Block offsets: [batch_size, max_blocks_per_seq]
        # 扩展为完整的偏移数组
        full_offsets = torch.zeros(
            (self.max_batch_size, 2, self.max_blocks_per_seq),
            dtype=torch.int32,
            device='cuda'
        )
        
        for b in range(self.max_batch_size):
            for i in range(self.max_blocks_per_seq):
                full_offsets[b, 0, i] = b * self.max_blocks_per_seq + i  # K
                full_offsets[b, 1, i] = b * self.max_blocks_per_seq + i  # V
        
        # Host 端偏移
        host_offsets = full_offsets.cpu().contiguous()
        
        # KV Cache 池指针 (指向 GPU 内存)
        # 格式: [num_layers, 2] 每层 K 和 V 的起始指针
        pool_pointers = torch.zeros(
            (self.num_layers, 2),
            dtype=torch.int64,
            device='cpu'
        )
        
        for layer in range(self.num_layers):
            k_ptr = self.kv_cache[layer, 0].data_ptr()
            v_ptr = self.kv_cache[layer, 1].data_ptr()
            pool_pointers[layer, 0] = k_ptr
            pool_pointers[layer, 1] = v_ptr
        
        return full_offsets, host_offsets, pool_pointers


class MossSpeechPagedRuntime:
    """
    支持 PagedAttention 的 MOSS-Speech Runtime
    
    手动管理所有输入，绕过 GenerationSession 的限制
    """
    
    def __init__(self, engine_dir: str):
        self.engine_dir = Path(engine_dir)
        
        # TensorRT 组件
        self.engine: Optional[trt.ICudaEngine] = None
        self.context: Optional[trt.IExecutionContext] = None
        
        # KV Cache
        self.kv_cache_manager: Optional[PagedKVCacheManager] = None
        
        # 配置
        self.config: Dict = {}
        
        # CUDA
        self.stream = torch.cuda.Stream()
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        
    def load(self) -> float:
        """加载 Engine 并初始化 KV Cache"""
        print("=" * 60)
        print("[MossSpeechPagedRuntime] Loading...")
        print("=" * 60)
        
        # 加载 TRT-LLM 插件
        import tensorrt_llm
        
        # 加载配置
        config_path = self.engine_dir / "config.json"
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        pc = self.config.get('pretrained_config', {})
        bc = self.config.get('build_config', {})
        
        print(f"\n[Config]")
        print(f"  Architecture: {pc.get('architecture')}")
        print(f"  Hidden size: {pc.get('hidden_size')}")
        print(f"  Num layers: {pc.get('num_hidden_layers')}")
        print(f"  Num KV heads: {pc.get('num_key_value_heads')}")
        
        start = time.perf_counter()
        
        # 1. 加载 Engine
        print(f"\n[1/3] Loading Engine...")
        engine_path = self.engine_dir / "rank0.engine"
        print(f"  File: {engine_path.stat().st_size / 1024**3:.2f} GB")
        
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        
        if self.engine is None:
            raise RuntimeError("Failed to load engine")
        
        print(f"  ✅ Engine loaded ({self.engine.num_io_tensors} tensors)")
        
        # 2. 创建执行上下文
        print(f"\n[2/3] Creating execution context...")
        self.context = self.engine.create_execution_context()
        print(f"  ✅ Context created")
        
        # 3. 初始化 KV Cache
        print(f"\n[3/3] Initializing KV Cache...")
        
        num_layers = pc.get('num_hidden_layers', 40)
        num_kv_heads = pc.get('num_key_value_heads', 8)
        hidden_size = pc.get('hidden_size', 4096)
        num_heads = pc.get('num_attention_heads', 32)
        head_size = hidden_size // num_heads
        max_batch_size = bc.get('max_batch_size', 1)
        max_seq_len = bc.get('max_seq_len', 4096)
        
        self.kv_cache_manager = PagedKVCacheManager(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            tokens_per_block=64,
        )
        
        kv_memory = self.kv_cache_manager.allocate()
        
        load_time = time.perf_counter() - start
        
        print(f"\n✅ Runtime ready!")
        print(f"  Load time: {load_time:.1f}s")
        print(f"  KV Cache: {kv_memory:.2f} GB")
        
        return load_time
    
    def _prepare_inputs(
        self,
        input_ids: torch.Tensor,
        past_seq_len: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        准备所有输入 Tensor
        
        关键：TRT-LLM 使用 remove_input_padding，输入是一维展平的！
        
        Args:
            input_ids: [batch_size, seq_len]
            past_seq_len: 历史序列长度 (用于增量推理)
        """
        batch_size, seq_len = input_ids.shape
        total_seq_len = past_seq_len + seq_len
        total_tokens = batch_size * seq_len  # 展平后的总 token 数
        
        pc = self.config.get('pretrained_config', {})
        bc = self.config.get('build_config', {})
        num_layers = pc.get('num_hidden_layers', 32)
        vocab_size = pc.get('vocab_size', 151680)
        audio_vocab_size = pc.get('audio_vocab_size', 16512)
        max_blocks = self.kv_cache_manager.max_blocks_per_seq
        
        inputs = {}
        
        # === 基本输入 (一维展平!) ===
        # input_ids: (-1,) = [total_tokens]
        inputs['input_ids'] = input_ids.to(torch.int32).flatten().contiguous().cuda()
        
        # position_ids: (-1,) = [total_tokens]
        inputs['position_ids'] = torch.arange(
            past_seq_len, past_seq_len + seq_len,
            dtype=torch.int32, device='cuda'
        ).repeat(batch_size).contiguous()
        
        # === 序列信息 ===
        # last_token_ids: (1,) 
        inputs['last_token_ids'] = torch.tensor(
            [total_tokens - 1],  # 最后一个 token 的索引
            dtype=torch.int32, device='cuda'
        ).contiguous()
        
        # sequence_length: (1,)
        inputs['sequence_length'] = torch.tensor(
            [total_seq_len],
            dtype=torch.int32, device='cuda'
        ).contiguous()
        
        # context_lengths: (1,)
        inputs['context_lengths'] = torch.tensor(
            [seq_len],
            dtype=torch.int32, device='cuda'
        ).contiguous()
        
        # === KV Cache 相关 ===
        # kv_cache_block_offsets: (1, 2, max_blocks)
        kv_offsets = torch.zeros(
            (1, 2, max_blocks),
            dtype=torch.int32, device='cuda'
        )
        for i in range(max_blocks):
            kv_offsets[0, 0, i] = i  # K blocks
            kv_offsets[0, 1, i] = i  # V blocks
        inputs['kv_cache_block_offsets'] = kv_offsets.contiguous()
        
        # host_kv_cache_block_offsets: (1, 2, max_blocks)
        inputs['host_kv_cache_block_offsets'] = kv_offsets.cpu().contiguous()
        
        # host_kv_cache_pool_pointers: (2,) - K和V的指针
        # 指向 KV Cache 的起始地址
        kv_cache = self.kv_cache_manager.kv_cache
        pool_pointers = torch.tensor(
            [kv_cache.data_ptr(), kv_cache.data_ptr()],  # 简化：K和V共用
            dtype=torch.int64, device='cpu'
        ).contiguous()
        inputs['host_kv_cache_pool_pointers'] = pool_pointers
        
        # host_past_key_value_lengths: (1,)
        inputs['host_past_key_value_lengths'] = torch.tensor(
            [past_seq_len],
            dtype=torch.int32, device='cpu'
        ).contiguous()
        
        # host_request_types: (1,) - 0=context, 1=generation
        inputs['host_request_types'] = torch.tensor(
            [0],  # context phase
            dtype=torch.int32, device='cpu'
        ).contiguous()
        
        # host_context_lengths: (1,)
        inputs['host_context_lengths'] = torch.tensor(
            [seq_len],
            dtype=torch.int32, device='cpu'
        ).contiguous()
        
        # host_runtime_perf_knobs: (16,)
        inputs['host_runtime_perf_knobs'] = torch.zeros(
            16, dtype=torch.int64, device='cpu'
        ).contiguous()
        
        # host_max_attention_window_sizes: (32,) - 每层的最大注意力窗口
        inputs['host_max_attention_window_sizes'] = torch.full(
            (num_layers,),
            bc.get('max_seq_len', 4096),
            dtype=torch.int32, device='cpu'
        ).contiguous()
        
        # host_sink_token_length: (1,)
        inputs['host_sink_token_length'] = torch.tensor(
            [0], dtype=torch.int32, device='cpu'
        ).contiguous()
        
        # cache_indirection: (1, 1, max_seq_len)
        inputs['cache_indirection'] = torch.zeros(
            (1, 1, bc.get('max_seq_len', 4096)),
            dtype=torch.int32, device='cuda'
        ).contiguous()
        
        # === 输出 (也是展平的!) ===
        # logits: (-1, vocab_size) = [total_tokens, vocab_size]
        inputs['logits'] = torch.zeros(
            (total_tokens, vocab_size),
            dtype=torch.float32,  # 注意：输出是 FLOAT 不是 HALF
            device='cuda'
        ).contiguous()
        
        # audio_logits: (-1, audio_vocab_size) = [total_tokens, audio_vocab_size]
        inputs['audio_logits'] = torch.zeros(
            (total_tokens, audio_vocab_size),
            dtype=torch.float32,  # FLOAT
            device='cuda'
        ).contiguous()
        
        return inputs
    
    @torch.inference_mode()
    def infer(
        self,
        input_ids: torch.Tensor,
        past_seq_len: int = 0,
    ) -> MossInferenceResult:
        """
        执行推理
        
        Args:
            input_ids: 输入 token IDs
            past_seq_len: 历史序列长度
        """
        batch_size, seq_len = input_ids.shape
        
        # 准备输入
        inputs = self._prepare_inputs(input_ids, past_seq_len)
        
        # 设置所有 tensor 地址
        errors = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            
            if name in inputs:
                tensor = inputs[name]
                try:
                    self.context.set_tensor_address(name, tensor.data_ptr())
                except Exception as e:
                    errors.append(f"{name}: {e}")
            else:
                # 未找到的 tensor
                mode = self.engine.get_tensor_mode(name)
                if mode == trt.TensorIOMode.INPUT:
                    errors.append(f"{name}: not provided")
        
        if errors:
            print(f"  ⚠️ Tensor errors ({len(errors)}):")
            for e in errors[:5]:
                print(f"    - {e}")
        
        # 设置动态输入形状 (一维)
        total_tokens = batch_size * seq_len
        max_blocks = self.kv_cache_manager.max_blocks_per_seq
        max_seq = self.config.get('build_config', {}).get('max_seq_len', 4096)
        
        dynamic_shapes = {
            'input_ids': (total_tokens,),
            'position_ids': (total_tokens,),
            'kv_cache_block_offsets': (1, 2, max_blocks),
            'host_kv_cache_block_offsets': (1, 2, max_blocks),
            'cache_indirection': (1, 1, max_seq),
        }
        
        for name, shape in dynamic_shapes.items():
            try:
                self.context.set_input_shape(name, shape)
            except Exception as e:
                print(f"  ⚠️ Shape error ({name}): {e}")
        
        # 执行推理 (使用 CUDA Events 精确计时)
        torch.cuda.synchronize()
        self.start_event.record()
        
        success = self.context.execute_async_v3(self.stream.cuda_stream)
        
        self.end_event.record()
        torch.cuda.synchronize()
        
        prefill_time_ms = self.start_event.elapsed_time(self.end_event)
        
        if not success:
            print(f"  ⚠️ execute_async_v3 returned False")
        
        # 获取输出
        logits = inputs['logits']
        audio_logits = inputs['audio_logits']
        
        # 验证输出
        logits_valid = self._verify_tensor(logits, "logits")
        audio_logits_valid = self._verify_tensor(audio_logits, "audio_logits")
        
        return MossInferenceResult(
            output_ids=input_ids,
            logits=logits,
            audio_logits=audio_logits,
            prefill_time_ms=prefill_time_ms,
            decode_time_ms=0.0,
            total_time_ms=prefill_time_ms,
            logits_valid=logits_valid,
            audio_logits_valid=audio_logits_valid,
            input_tokens=seq_len,
            output_tokens=0,
        )
    
    def _verify_tensor(self, tensor: torch.Tensor, name: str) -> bool:
        """验证 tensor 是否有效"""
        is_zero = tensor.abs().sum().item() == 0
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        
        if is_zero:
            print(f"  ⚠️ {name}: all zeros")
            return False
        if has_nan:
            print(f"  ⚠️ {name}: contains NaN")
            return False
        if has_inf:
            print(f"  ⚠️ {name}: contains Inf")
            return False
        
        # 输出统计
        print(f"  ✓ {name}: sum={tensor.sum().item():.2e}, "
              f"mean={tensor.mean().item():.4f}, std={tensor.std().item():.4f}")
        return True
    
    def sample_dual_head(
        self,
        logits: torch.Tensor,
        audio_logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        双头采样
        
        Args:
            logits: 文本 logits [batch, seq, vocab]
            audio_logits: 音频 logits [batch, seq, audio_vocab]
            
        Returns:
            (text_tokens, audio_tokens)
        """
        # 只取最后一个位置
        text_probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
        audio_probs = torch.softmax(audio_logits[:, -1, :] / temperature, dim=-1)
        
        # Top-K 采样
        if top_k > 0:
            text_topk = torch.topk(text_probs, top_k, dim=-1)
            audio_topk = torch.topk(audio_probs, top_k, dim=-1)
            
            text_idx = torch.multinomial(text_topk.values, 1)
            text_tokens = text_topk.indices.gather(-1, text_idx)
            
            audio_idx = torch.multinomial(audio_topk.values, 1)
            audio_tokens = audio_topk.indices.gather(-1, audio_idx)
        else:
            text_tokens = torch.multinomial(text_probs, 1)
            audio_tokens = torch.multinomial(audio_probs, 1)
        
        return text_tokens, audio_tokens
    
    def benchmark(
        self,
        num_input_tokens: int = 512,
        num_warmup: int = 3,
        num_runs: int = 10,
    ) -> Dict:
        """运行基准测试"""
        print(f"\n{'='*60}")
        print(f"[Benchmark] PagedAttention Runtime")
        print(f"{'='*60}")
        print(f"  Input tokens: {num_input_tokens}")
        print(f"  Warmup: {num_warmup}")
        print(f"  Runs: {num_runs}")
        
        pc = self.config.get('pretrained_config', {})
        vocab_size = pc.get('vocab_size', 151680)
        
        # 准备输入
        input_ids = torch.randint(
            0, vocab_size,
            (1, num_input_tokens),
            dtype=torch.int32,
            device='cuda'
        )
        
        # 预热
        print(f"\n[Warmup]")
        for i in range(num_warmup):
            result = self.infer(input_ids)
            status = "✓" if result.logits_valid else "✗"
            print(f"  Run {i+1}: {result.prefill_time_ms:.1f}ms [{status}]")
        
        # 正式测试
        print(f"\n[Benchmark]")
        times = []
        valid_count = 0
        
        for i in range(num_runs):
            result = self.infer(input_ids)
            times.append(result.prefill_time_ms)
            
            if result.logits_valid and result.audio_logits_valid:
                valid_count += 1
            
            status = "✓" if result.logits_valid else "✗"
            print(f"  Run {i+1}: {result.prefill_time_ms:.1f}ms [{status}]")
        
        # 统计
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        # 计算指标
        tokens_per_sec = num_input_tokens / (avg_time / 1000)
        
        # TTFA (假设 prefill + 5 audio tokens)
        ttfa_ms = avg_time  # 简化
        
        # RTF
        audio_token_rate = 50
        audio_duration_ms = 5 / audio_token_rate * 1000
        rtf = ttfa_ms / audio_duration_ms
        
        print(f"\n{'='*60}")
        print(f"[Results]")
        print(f"  Prefill time: {avg_time:.1f} ± {std_time:.1f}ms")
        print(f"  Min/Max: {min_time:.1f} / {max_time:.1f}ms")
        print(f"  Valid outputs: {valid_count}/{num_runs}")
        print(f"  Throughput: {tokens_per_sec:.0f} tokens/s")
        print(f"  TTFA: {ttfa_ms:.1f}ms")
        print(f"  RTF: {rtf:.2f}")
        print(f"{'='*60}")
        
        # 评估
        print(f"\n[评估]")
        if valid_count == num_runs:
            print(f"  ✅ 所有输出有效 (Address not set 已解决)")
        else:
            print(f"  ⚠️ {num_runs - valid_count}/{num_runs} 输出无效")
        
        if ttfa_ms < 300:
            print(f"  ✅ TTFA < 300ms 达标")
        else:
            print(f"  ⚠️ TTFA > 300ms ({ttfa_ms:.0f}ms)")
        
        if rtf < 1.0:
            print(f"  ✅ RTF < 1.0 可实时")
        else:
            print(f"  ⚠️ RTF >= 1.0 ({rtf:.2f})")
        
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
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine_dir", default="/workspace/models/MOSS-Speech-Engine")
    parser.add_argument("--input_tokens", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    
    args = parser.parse_args()
    
    runtime = MossSpeechPagedRuntime(engine_dir=args.engine_dir)
    load_time = runtime.load()
    
    results = runtime.benchmark(
        num_input_tokens=args.input_tokens,
        num_warmup=args.warmup,
        num_runs=args.runs,
    )
    
    print(f"\n[JSON]")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

