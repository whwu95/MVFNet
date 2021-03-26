"""model builder"""
import torch.nn as nn

from ..utils import Registry, build_from_cfg

RECOGNIZERS = Registry('recognizer')
BACKBONES = Registry('backbone')
HEADS = Registry('head')
SPATIAL_TEMPORAL_MODULES = Registry('spatial_temporal_module')
SEGMENTAL_CONSENSUSES = Registry('segmental_consensus')


def build(cfg, registry, default_args=None):
    """build model for config dict"""
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_recognizer(cfg, train_cfg=None, test_cfg=None):
    """build recognizer"""
    return build(cfg, RECOGNIZERS,
                 dict(train_cfg=train_cfg, test_cfg=test_cfg))


def build_backbone(cfg):
    """build backbone"""
    return build(cfg, BACKBONES)


def build_head(cfg):
    """build head"""
    return build(cfg, HEADS)


def build_spatial_temporal_module(cfg):
    """build st module"""
    return build(cfg, SPATIAL_TEMPORAL_MODULES)


def build_segmental_consensus(cfg):
    """build consensus"""
    return build(cfg, SEGMENTAL_CONSENSUSES)
