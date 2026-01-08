#!/usr/bin/env python3
"""
快速测试脚本 - 无需加载模型
用于验证流式传输逻辑、API 结构等

运行方式:
    python test_stream_mock.py
"""

import asyncio
import json
import time
from typing import AsyncGenerator

# Mock Brain Handler - 模拟真实的流式输出
class MockBrainHandler:
    def __init__(self):
        self.is_ready = True
    
    async def generate_stream(self, request: dict) -> AsyncGenerator[str, None]:
        """模拟流式生成 - 每个 token 延迟 50ms"""
        prompt = request.get("prompt", "")
        print(f"[Mock] Received prompt: {prompt[:50]}...")
        
        # 模拟回复
        response = "你好！我是Trinity，一个温柔的数字女友。有什么我可以帮助你的吗？"
        
        # 逐字输出，模拟真实流式
        for char in response:
            await asyncio.sleep(0.05)  # 50ms per token
            yield char
        
        print(f"[Mock] Done, total chars: {len(response)}")

# 测试 SSE 流式传输
async def test_sse_stream():
    print("=" * 50)
    print("测试 1: SSE 流式传输")
    print("=" * 50)
    
    brain = MockBrainHandler()
    request = {"prompt": "你好，介绍一下你自己"}
    
    start_time = time.time()
    first_token_time = None
    tokens = []
    
    async for token in brain.generate_stream(request):
        if first_token_time is None:
            first_token_time = time.time()
            ttft = (first_token_time - start_time) * 1000
            print(f"⚡ TTFT (Time to First Token): {ttft:.2f}ms")
        
        tokens.append(token)
        print(token, end="", flush=True)
    
    print()
    total_time = (time.time() - start_time) * 1000
    print(f"\n✅ 总耗时: {total_time:.2f}ms")
    print(f"✅ 总字符: {len(tokens)}")
    print(f"✅ 速度: {len(tokens) / (total_time / 1000):.2f} chars/s")

# 测试 HTTP 端点结构
async def test_api_structure():
    print("\n" + "=" * 50)
    print("测试 2: API 结构验证")
    print("=" * 50)
    
    # 模拟 SSE 事件格式
    brain = MockBrainHandler()
    request = {"prompt": "test", "max_tokens": 50}
    
    events = []
    async for token in brain.generate_stream(request):
        event = f"data: {json.dumps({'token': token})}\n\n"
        events.append(event)
    
    # 添加结束标记
    events.append(f"data: {json.dumps({'done': True})}\n\n")
    
    print(f"✅ 生成了 {len(events)} 个 SSE 事件")
    print(f"✅ 示例事件: {events[0].strip()}")
    print(f"✅ 结束事件: {events[-1].strip()}")

# 测试延迟要求
async def test_latency_requirement():
    print("\n" + "=" * 50)
    print("测试 3: 延迟要求验证 (目标 <200ms TTFT)")
    print("=" * 50)
    
    brain = MockBrainHandler()
    
    # 模拟 10 次请求
    ttfts = []
    for i in range(5):
        request = {"prompt": f"测试请求 {i}"}
        start = time.time()
        
        async for token in brain.generate_stream(request):
            ttft = (time.time() - start) * 1000
            ttfts.append(ttft)
            break  # 只计算第一个 token
    
    avg_ttft = sum(ttfts) / len(ttfts)
    print(f"✅ 平均 TTFT: {avg_ttft:.2f}ms")
    
    if avg_ttft < 200:
        print("🎉 满足 <200ms 延迟要求!")
    else:
        print(f"⚠️ 超出延迟要求 ({avg_ttft:.2f}ms > 200ms)")

async def main():
    print("🧪 Project Trinity - 快速测试 (Mock 模式)")
    print("无需加载模型，用于验证代码逻辑\n")
    
    await test_sse_stream()
    await test_api_structure()
    await test_latency_requirement()
    
    print("\n" + "=" * 50)
    print("✅ 所有测试完成!")
    print("如果逻辑正确，再运行 ./run_microservices.sh 加载真实模型")
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(main())

