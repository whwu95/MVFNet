"""base recognizer"""
from abc import ABCMeta, abstractmethod

import torch.nn as nn
import torch.nn.functional as F

from ...core import auto_fp16
from ..builder import build_backbone, build_head


class BaseRecognizer(nn.Module, metaclass=ABCMeta):
    """Abstract base class for recognizers"""

    def __init__(self, backbone, cls_head):
        super(BaseRecognizer, self).__init__()
        self.fp16_enabled = False
        self.backbone = build_backbone(backbone)
        if cls_head is not None:
            self.cls_head = build_head(cls_head)
        self.init_weights()

    @property
    def with_cls_head(self):
        return hasattr(self, 'cls_head') and self.cls_head is not None

    @abstractmethod
    def forward_train(self, imgs, label, **kwargs):
        pass

    @abstractmethod
    def forward_test(self, imgs, **kwargs):
        pass

    def init_weights(self):
        self.backbone.init_weights()
        if self.with_cls_head:
            self.cls_head.init_weights()

    def extract_feat(self, img_group):
        x = self.backbone(img_group)
        return x

    def average_clip(self, cls_score):
        """Averaging class score over multiple clips.

        Using different averaging types ('score' or 'prob' or None,
        which defined in test_cfg) to computed the final averaged
        class score.

        Args:
            cls_score (torch.Tensor): Class score to be averaged.

        return:
            torch.Tensor: Averaged class score.
        """
        if self.test_cfg is None:
            self.test_cfg = {}
            self.test_cfg['average_clips'] = None

        if 'average_clips' not in self.test_cfg.keys():
            # self.test_cfg['average_clips'] = None
            raise KeyError('"average_clips" must defined in test_cfg\'s keys')

        average_clips = self.test_cfg['average_clips']
        if average_clips not in ['score', 'prob', None]:
            raise ValueError(f'{average_clips} is not supported. '
                             f'Currently supported ones are '
                             f'["score", "prob", None]')

        if average_clips == 'prob':
            cls_score = F.softmax(cls_score, dim=1).mean(dim=0, keepdim=True)
        elif average_clips == 'score':
            cls_score = cls_score.mean(dim=0, keepdim=True)
        return cls_score

    @auto_fp16(apply_to=('img_group', ))
    def forward(self, img_group, label, return_loss=True,
                return_numpy=True, **kwargs):
        if return_loss:
            return self.forward_train(img_group, label, **kwargs)
        else:
            return self.forward_test(img_group, return_numpy, **kwargs)
