import sys
import os
import time
import asyncio
from loguru import logger

# 添加路径
sys.path.insert(0, "/workspace/project-trinity/project-trinity")
sys.path.insert(0, "/workspace/project-trinity/project-trinity/server")
sys.path.insert(0, "/workspace/CosyVoice")
sys.path.insert(0, "/workspace/CosyVoice/third_party/Matcha-TTS")

from server.cortex.models.mouth import MouthHandler

async def test_vllm_inference():
    logger.info("🚀 开始 vLLM 独立测试 (无 Uvicorn)...")
    
    mouth = MouthHandler()
    
    # 初始化 (这会触发 load_vllm)
    logger.info("正在初始化 MouthHandler (启用 vLLM)...")
    success = await mouth.initialize()
    
    if not success:
        logger.error("❌ 初始化失败")
        return

    logger.info("✅ 初始化成功！准备进行推理...")
    
    # 预热 / 推理
    text = "你好呀，我是小星。"
    logger.info(f"正在合成: {text}")
    
    start_time = time.time()
    count = 0
    first_token_time = None
    
    try:
        # 使用流式接口
        async for chunk in mouth.synthesize_stream(text):
            if not first_token_time:
                first_token_time = time.time()
                ttft = first_token_time - start_time
                logger.success(f"⚡ TTFT: {ttft:.4f}s")
            count += 1
            if count % 10 == 0:
                logger.info(f"收到第 {count} 个音频块")
                
        total_time = time.time() - start_time
        logger.success(f"🎉 合成完成！总耗时: {total_time:.4f}s")
        
    except Exception as e:
        logger.error(f"❌ 推理过程出错: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(test_vllm_inference())








