"""consensus for 2D cls head (used for multiple frames)"""
from .relation_consensus import return_TRN
from .simple_consensus import SimpleConsensus

__all__ = [
    'SimpleConsensus',
    'return_TRN'
]
