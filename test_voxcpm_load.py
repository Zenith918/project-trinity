"""
VoxCPM 1.5 Loading Test - Correct Way
"""
import sys
import os
from loguru import logger

# Set environment to avoid downloading
os.environ["TRANSFORMERS_OFFLINE"] = "0"  # Allow download if needed

logger.info("Testing VoxCPM 1.5 loading...")

try:
    logger.info("Step 1: Importing voxcpm...")
    from voxcpm import VoxCPM
    logger.success("Import successful!")
    
    # Method 1: Use from_pretrained with HF model ID (will use cache if available)
    logger.info("Step 2: Loading model from HuggingFace (openbmb/VoxCPM1.5)...")
    logger.info("This may take a few minutes on first run...")
    
    model = VoxCPM.from_pretrained(
        hf_model_id="openbmb/VoxCPM1.5",
        load_denoiser=False,  # Skip denoiser to speed up loading
        optimize=True,
        local_files_only=False  # Allow downloading if not cached
    )
    logger.success("Model loaded successfully!")
    
    # Quick test: generate a short audio
    logger.info("Step 3: Testing generation...")
    test_text = "Hello, this is a test."
    audio = model.generate(test_text, prompt_text="Hi", prompt_wav_path=None)
    logger.success(f"Generation test passed! Audio shape: {audio.shape if hasattr(audio, 'shape') else 'OK'}")
    
except Exception as e:
    logger.error(f"Failed: {e}")
    import traceback
    logger.error(traceback.format_exc())
