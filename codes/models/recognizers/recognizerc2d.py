""" RecognizerC2D for 2D backbone
tensor shape BxCxTxHxW instead of BTxCHW (recognizer2d) 
backbone forward: need BxCxTxHxW tensor
module forward: need BxCxTxHxW tensor (don't need to set n_segment)
"""

import torch.nn as nn
from torch.nn.functional import softmax

from ..builder import RECOGNIZERS
from .base import BaseRecognizer


@RECOGNIZERS.register_module
class RecognizerC2D(BaseRecognizer):
    """class for recognizerc2d"""
    def __init__(self,
                 modality='RGB',
                 backbone='BNInception',
                 cls_head='I3DClsHead',
                 fcn_testing=False,
                 module_cfg=None,
                 nonlocal_cfg=None,
                 train_cfg=None,
                 test_cfg=None):
        super(RecognizerC2D, self).__init__(backbone, cls_head)
        self.fcn_testing = fcn_testing
        self.modality = modality
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.module_cfg = module_cfg

        # insert module into backbone
        if self.module_cfg:
            self._prepare_base_model(backbone, self.module_cfg, nonlocal_cfg)

        assert modality in ['RGB', 'Flow', 'RGBDiff']
        if modality in ['Flow', 'RGBDiff']:
            length = 5
            if modality == 'Flow':
                self.in_channels = 2 * length
            elif modality == 'RGBDiff':
                self.in_channels = 3 * length
            self._construct_2d_backbone_conv1(self.in_channels)
        elif modality == 'RGB':
            self.in_channels = 3
        else:
            raise ValueError(
                'Not implemented yet; modality should be '
                '["RGB", "RGBDiff", "Flow"]')

    def _construct_2d_backbone_conv1(self, in_channels):
        raise NotImplementedError


    def _prepare_base_model(self, backbone, module_cfg, nonlocal_cfg):
        # module_cfg
        # tsm: dict(type='tsm', n_frames=8 , n_div=8,
        #           shift_place='blockres',
        #           temporal_pool=False, two_path=False)
        # nolocal: dict(n_segment=8)

        backbone_name = backbone['type']
        module_name = module_cfg.pop('type')
        self.module_name = module_name
        if backbone_name == 'ResNet_I3D':
            if module_name == 'tsm':
                print('Adding temporal shift...')
                from ..modules.tsm_c2d import make_temporal_shift
                make_temporal_shift(self.backbone, **module_cfg)



    def forward_train(self, imgs, labels, **kwargs):
        """test"""
        #  imgs: [B 1 C T H W]
        #  imgs: [B C T H W]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])

        x = self.extract_feat(imgs)
        losses = dict()
        if self.with_cls_head:
            cls_score = self.cls_head(x)
            gt_label = labels.squeeze()
            loss_cls = self.cls_head.loss(cls_score, gt_label)
            losses.update(loss_cls)

        return losses

    def forward_test(self, imgs, return_numpy, **kwargs):
        """test"""
        #  imgs: [B Clips C T H W]
        #  imgs: [BxClips C T H W]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])

        x = self.extract_feat(imgs)

        if self.with_cls_head:
            cls_score = self.cls_head(x)
            cls_score = self.average_clip(cls_score)

        if return_numpy:
            return cls_score.cpu().numpy()
        else:
            # for dp model gpu gather
            return cls_score
