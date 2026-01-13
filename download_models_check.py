import os
from huggingface_hub import snapshot_download

# 设置镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def download_model(repo_id, local_dir):
    print(f"📥 Downloading {repo_id} to {local_dir}...")
    try:
        # 先下载非权重文件
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            ignore_patterns=["*.bin", "*.safetensors", "*.pt", "*.pth"],
            local_dir_use_symlinks=False
        )
        print(f"✅ Config/Code Downloaded: {local_dir}")
        
        # 检查是否有推理代码
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                if file.endswith(".py") or file.endswith(".md"):
                    print(f"   - {file}")
                    
    except Exception as e:
        print(f"❌ Failed: {e}")

if __name__ == "__main__":
    download_model("openbmb/VoxCPM1.5", "/workspace/models/VoxCPM1.5")
    download_model("IndexTeam/IndexTTS-2", "/workspace/models/IndexTTS2.5")







