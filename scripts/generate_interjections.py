import asyncio
import os
import sys
import aiohttp
import json
from loguru import logger

# 目标目录
OUTPUT_DIR = "/workspace/project-trinity/project-trinity/assets/interjections"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 插话清单
INTERJECTIONS = {
    "hmm": {"text": "嗯...", "instruct": "用思考、犹豫的语气说"},
    "yeah": {"text": "嗯嗯！", "instruct": "用开心、赞同的语气说"},
    "wow": {"text": "哇！", "instruct": "用惊讶、惊喜的语气说"},
    "laugh": {"text": "哈哈", "instruct": "用开心的笑声说"},
    "sigh": {"text": "唉...", "instruct": "用遗憾、叹气的语气说"},
    "wait": {"text": "让我想想...", "instruct": "用思考的语气说"},
}

async def generate_one(name, config):
    url = "http://localhost:9001/tts"
    payload = {
        "text": config["text"],
        "instruct_text": config["instruct"],
        "stream": False
    }
    
    logger.info(f"🎙️ 生成: {name} - {config['text']}")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                if resp.status == 200:
                    content = await resp.read()
                    file_path = os.path.join(OUTPUT_DIR, f"{name}.wav")
                    with open(file_path, "wb") as f:
                        f.write(content)
                    logger.success(f"✅ 保存: {file_path}")
                else:
                    logger.error(f"❌ 失败 {name}: {resp.status}")
    except Exception as e:
        logger.error(f"❌ 错误 {name}: {e}")

async def main():
    logger.info("⏳ 等待 Mouth 服务就绪...")
    # 简单的轮询等待
    for _ in range(30):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:9001/health") as resp:
                    if resp.status == 200:
                        logger.info("Mouth 服务已就绪！")
                        break
        except:
            await asyncio.sleep(2)
    else:
        logger.error("Mouth 服务未就绪，退出")
        return

    # 并发生成
    tasks = [generate_one(k, v) for k, v in INTERJECTIONS.items()]
    await asyncio.gather(*tasks)
    logger.success("🎉 所有插话音频生成完毕！")

if __name__ == "__main__":
    asyncio.run(main())







