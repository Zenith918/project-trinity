import os
import sys
import asyncio
from loguru import logger

# 设置环境
os.environ["TRINITY_DEBUG"] = "true"
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "server"))

# 强制添加 CosyVoice 路径
COSYVOICE_PATH = "/workspace/CosyVoice"
if os.path.exists(COSYVOICE_PATH) and COSYVOICE_PATH not in sys.path:
    sys.path.insert(0, COSYVOICE_PATH)
    
# 添加 Matcha
MATCHA_PATH = "/workspace/CosyVoice/third_party/Matcha-TTS"
if os.path.exists(MATCHA_PATH) and MATCHA_PATH not in sys.path:
    sys.path.insert(0, MATCHA_PATH)

from server.cortex.models.brain import BrainHandler
from server.cortex.models.mouth import MouthHandler

async def test_startup():
    logger.info("🛠️ 开始 Cortex 启动调试...")
    
    # 1. 初始化 Brain
    logger.info("--- 阶段 1: 初始化 Brain (vLLM) ---")
    brain = BrainHandler()
    try:
        success = await brain.initialize()
        if success:
            logger.success("✅ Brain 初始化成功")
        else:
            logger.error("❌ Brain 初始化失败")
    except Exception as e:
        logger.error(f"❌ Brain 抛出异常: {e}")
        
    # 2. 初始化 Mouth
    logger.info("--- 阶段 2: 初始化 Mouth (CosyVoice) ---")
    mouth = MouthHandler()
    try:
        success = await mouth.initialize()
        if success:
            logger.success("✅ Mouth 初始化成功")
        else:
            logger.error("❌ Mouth 初始化失败")
    except Exception as e:
        logger.error(f"❌ Mouth 抛出异常: {e}")
        import traceback
        logger.error(traceback.format_exc())

    logger.info("🏁 调试结束")

if __name__ == "__main__":
    asyncio.run(test_startup())

