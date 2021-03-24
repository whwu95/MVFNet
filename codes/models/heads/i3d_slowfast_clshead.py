"""slowfast head for 3D input (B C T H W)"""
from .i3d_clshead import I3DClsHead
import torch
import torch.nn as nn
from ..builder import HEADS

@HEADS.register_module
class I3DSlowFastClsHead(I3DClsHead):
    """slowfast head"""
    def forward(self, x):
        if not self.fcn_testing:
            # [30 2048 4 8 8]
            if isinstance(x, tuple):
                slow, fast = x
                slow = self.Logits(slow)  # [1, 2048, 4, 7, 7]
                fast = self.Logits(fast)
                x = torch.cat([slow, fast], dim=1)
            else:
                x = self.Logits(x)
 
            # [30 2048 1 1 1]
            if self.dropout is not None:
                x = self.dropout(x)
            # [30 2048 1 1 1]
            x = x.view(x.shape[0], -1)
            # [30 2048]
            cls_score = self.fc_cls(x)
            # [B*clip_num 400]   train:clip_num=1 test:b=1
        else:
            # [30 2048 4 8 8]
            if isinstance(x, tuple):
                slow, fast = x
                x = torch.cat([slow.mean(2, True), fast.mean(2, True)], dim=1)  # [1, 2048+256, 1, 8,8]
        
            if self.new_cls is None:
                self.new_cls = nn.Conv3d(
                    self.in_channels,
                    self.num_classes,
                    1, 1, 0).cuda()
                self.new_cls.load_state_dict(
                    {'weight': self.fc_cls.weight.unsqueeze(-1).unsqueeze(
                        -1).unsqueeze(-1),
                     'bias': self.fc_cls.bias})

            class_map = self.new_cls(x)
            # [30 400 4 8 8]
            cls_score = class_map.mean([2, 3, 4])
            # [30 400]
        return cls_score
