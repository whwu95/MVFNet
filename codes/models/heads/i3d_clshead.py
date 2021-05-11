"""cls head for 3D input (B C T H W)"""
import torch.nn as nn
from ..builder import HEADS
from .base import BaseHead
@HEADS.register_module
class I3DClsHead(BaseHead):
    """cls head for 3D input (B C T H W)"""

    def __init__(self,
                 spatial_type='avg',
                 spatial_size=7,
                 temporal_size=4,
                 consensus_cfg=dict(type='avg', dim=1),
                 dropout_ratio=0.5,
                 in_channels=2048,
                 num_classes=400,
                 init_std=0.01,
                 fcn_testing=False,
                 extract_feat=False,
                 ):
        super(I3DClsHead, self).__init__(spatial_size, dropout_ratio,
                                         in_channels, num_classes, init_std, extract_feat)
        self.spatial_type = spatial_type
        self.consensus_type = consensus_cfg['type']
        self.temporal_size = temporal_size
        assert not (self.spatial_size == -1) ^ (self.temporal_size == -1)
        if self.temporal_size == -1 and self.spatial_size == -1:
            self.pool_size = (1, 1, 1)
            if self.spatial_type == 'avg':
                self.Logits = nn.AdaptiveAvgPool3d(self.pool_size)
            if self.spatial_type == 'max':
                self.Logits = nn.AdaptiveMaxPool3d(self.pool_size)
        else:
            self.pool_size = (self.temporal_size, ) + self.spatial_size
            if self.spatial_type == 'avg':
                self.Logits = nn.AvgPool3d(
                    self.pool_size, stride=1, padding=0)
            if self.spatial_type == 'max':
                self.Logits = nn.MaxPool3d(
                    self.pool_size, stride=1, padding=0)
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)
        self.fcn_testing = fcn_testing
        self.new_cls = None

    def forward(self, x):
        """forward """
        if not self.fcn_testing:
            # [30 2048 4 8 8]
            x = self.Logits(x)
            # [30 2048 1 1 1]
            if self.dropout is not None:
                x = self.dropout(x)
            # [30 2048 1 1 1]
            x = x.view(x.shape[0], -1)
            # [30 2048]
            if self.extract_feat:
                cls_score = x  # [30 2048]
            else:
                cls_score = self.fc_cls(x)
            # [B*clip_num 400]   train:clip_num=1 test:b=1
        else:
            # [30 2048 4 8 8]
            if self.new_cls is None:
                self.new_cls = nn.Conv3d(
                    self.in_channels,
                    self.num_classes,
                    1, 1, 0).cuda()
                self.new_cls.load_state_dict(
                    {'weight': self.fc_cls.weight.unsqueeze(-1).unsqueeze(
                        -1).unsqueeze(-1),
                     'bias': self.fc_cls.bias})
            if self.extract_feat:
                class_map = x  # [30 2048 4 8 8]
            else:
                class_map = self.new_cls(x)
                # [30 400 4 8 8]
            cls_score = class_map.mean([2, 3, 4])
            # [30 400] or [30 feat-dim]
        return cls_score

    def init_weights(self):
        """init weights"""
        nn.init.normal_(self.fc_cls.weight, 0, self.init_std)
        nn.init.constant_(self.fc_cls.bias, 0)
