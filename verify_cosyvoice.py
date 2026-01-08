import os
import sys
from loguru import logger

# 强制添加 CosyVoice 路径
COSYVOICE_PATH = "/workspace/CosyVoice"
MATCHA_PATH = "/workspace/CosyVoice/third_party/Matcha-TTS"

if os.path.exists(COSYVOICE_PATH) and COSYVOICE_PATH not in sys.path:
    sys.path.insert(0, COSYVOICE_PATH)
if os.path.exists(MATCHA_PATH) and MATCHA_PATH not in sys.path:
    sys.path.insert(0, MATCHA_PATH)

def test_load_cosyvoice():
    model_path = "/workspace/models/CosyVoice3-0.5B"
    logger.info(f"🔍 开始验证 CosyVoice 3 加载: {model_path}")
    
    # 检查路径
    if not os.path.exists(model_path):
        logger.error(f"❌ 模型路径不存在: {model_path}")
        return False

    try:
        from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2, CosyVoice3
        logger.success("✅ CosyVoice 库导入成功")
        
        # 模拟修复逻辑
        config_v3 = os.path.join(model_path, "cosyvoice3.yaml")
        config_default = os.path.join(model_path, "cosyvoice.yaml")
        config_v2 = os.path.join(model_path, "cosyvoice2.yaml") # 新增 V2 软链
        
        if os.path.exists(config_v3):
            logger.info("Found cosyvoice3.yaml")
            if not os.path.exists(config_default):
                logger.warning("Missing cosyvoice.yaml, creating symlink...")
                os.symlink(config_v3, config_default)
                logger.success("✅ Symlink created: cosyvoice3.yaml -> cosyvoice.yaml")
            
            # 同时创建 cosyvoice2.yaml 软链
            if not os.path.exists(config_v2):
                logger.warning("Missing cosyvoice2.yaml, creating symlink for compat...")
                os.symlink(config_v3, config_v2)
                logger.success("✅ Symlink created: cosyvoice3.yaml -> cosyvoice2.yaml")

        
        logger.info("⚡ 尝试初始化模型 (CosyVoice3)...")
        # CosyVoice3 不支持 load_jit
        model = CosyVoice3(model_path, load_trt=False)
        logger.success("🎉🎉🎉 CosyVoice 3.0 (via CosyVoice3) 加载成功！")

        
        # 简单推理测试
        logger.info("🎤 正在进行推理测试...")
        try:
            res = model.inference_instruct(
                "你好，我是 Trinity。", 
                "用开心的声音说",
                None # speaker_embedding
            )
            if 'audio' in res:
                logger.success("✅ 推理成功，音频已生成")
            else:
                logger.warning("⚠️ 推理返回结果异常")
        except Exception as e:
            logger.error(f"❌ 推理失败: {e}")
            
        return True
        
    except ImportError as e:
        logger.error(f"❌ 库导入失败: {e}")
    except Exception as e:
        logger.error(f"❌ 加载失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    return False

if __name__ == "__main__":
    test_load_cosyvoice()

