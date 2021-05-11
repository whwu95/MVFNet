# !/usr/bin/env python3
from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseHead(nn.Module, metaclass=ABCMeta):
    def __init__(
        self,
        spatial_size=7,
        dropout_ratio=0.8,
        in_channels=1024,
        num_classes=101,
        init_std=0.001,
        extract_feat=False,
    ):
        super(BaseHead, self).__init__()
        self.spatial_size = spatial_size
        if spatial_size != -1:
            self.spatial_size = (spatial_size, spatial_size)
        self.dropout_ratio = dropout_ratio
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.Logits = None
        self.extract_feat = extract_feat

    @abstractmethod
    def forward(self, x):
        pass

    def init_weights(self):
        pass

    def loss(self, cls_score, labels):
        losses = dict()
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        losses['loss_cls'] = F.cross_entropy(cls_score, labels)
        return losses
