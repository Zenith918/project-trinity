# MOSS-Speech TensorRT-LLM Custom Model
# 基于研究员方案: 32层共享 + 4层分支 (text/audio)

from .model import MossSpeechForTRTLLM
from .convert import convert_moss_to_trtllm

__all__ = ['MossSpeechForTRTLLM', 'convert_moss_to_trtllm']



