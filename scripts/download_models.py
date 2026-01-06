import os
from huggingface_hub import snapshot_download
from modelscope import snapshot_download as ms_download

def download_models():
    models_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"📂 模型下载目录: {models_dir}")
    
    # 1. Qwen 3.0-VL (Brain) - HuggingFace
    print("\n⬇️ 正在下载 Qwen 3.0-VL (8B-Instruct)...")
    # 注意：Qwen 3.0 VL 目前 (2026) 还没有官方 AWQ 量化版，我们先下载 fp16/bf16 原版
    # 如果显存吃紧，后续我们自己做量化，或者找社区量化版
    qwen_path = os.path.join(models_dir, "Qwen3-VL-8B-Instruct")
    try:
        snapshot_download(
            repo_id="Qwen/Qwen3-VL-8B-Instruct",
            local_dir=qwen_path,
            local_dir_use_symlinks=False,  # 确保是真实文件
            resume_download=True
        )
        print(f"✅ Qwen 3.0 下载完成: {qwen_path}")
    except Exception as e:
        print(f"❌ Qwen 3.0 下载失败: {e}")


    # 2. SenseVoice (Ears) - ModelScope (通常国内下载更快，或者用 HF)
    print("\n⬇️ 正在下载 SenseVoiceSmall...")
    sense_path = os.path.join(models_dir, "SenseVoiceSmall")
    try:
        # 使用 ModelScope 下载，因为它在中文语音方面通常更好
        ms_download(
            "iic/SenseVoiceSmall",
            local_dir=sense_path
        )
        print(f"✅ SenseVoice 下载完成: {sense_path}")
    except Exception as e:
        print(f"❌ SenseVoice 下载失败: {e}")

    # 3. GeneFace++ (Driver)
    # GeneFace 需要从特定的 Drive/Repo 下载，这里先创建目录
    geneface_path = os.path.join(models_dir, "geneface")
    os.makedirs(geneface_path, exist_ok=True)
    print(f"\n⚠️ GeneFace++ 模型需要手动放置到: {geneface_path}")
    print("   (通常需要从其 GitHub Release 或 Google Drive 下载)")

if __name__ == "__main__":
    download_models()


