"""
Project Trinity - Mind Engine
三位一体心智引擎

Layer 1: BioState (本我) - 概率内稳态与反射
Layer 2: NarrativeManager (超我) - 记忆与人设约束
Layer 3: EgoDirector (自我) - 决策与仲裁
"""

from .bio_state import BioState, BioStateSnapshot
from .narrative_mgr import NarrativeManager
from .ego_director import EgoDirector

__all__ = [
    "BioState",
    "BioStateSnapshot",
    "NarrativeManager",
    "EgoDirector"
]

