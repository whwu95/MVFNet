"""init"""
from .base import BaseHead
from .i3d_clshead import I3DClsHead
from .tsn_clshead import TSNClsHead
from .i3d_slowfast_clshead import I3DSlowFastClsHead

__all__ = ['BaseHead', 'TSNClsHead', 'I3DClsHead']
