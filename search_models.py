from huggingface_hub import HfApi
import sys

api = HfApi()

def search_model(keyword):
    print(f"🔍 Searching for '{keyword}'...")
    try:
        models = api.list_models(search=keyword, limit=5)
        found = False
        for model in models:
            print(f"  - {model.modelId} ({model.downloads} downloads)")
            found = True
        if not found:
            print(f"  ❌ No models found for '{keyword}'")
    except Exception as e:
        print(f"  ❌ Error: {e}")

if __name__ == "__main__":
    search_model("VoxCPM")
    search_model("IndexTTS")
    search_model("OpenBMB")
    search_model("IndexTeam")








