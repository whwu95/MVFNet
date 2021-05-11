"""recognizer2d"""
import torch.nn as nn
from ..builder import RECOGNIZERS
from .base import BaseRecognizer


@RECOGNIZERS.register_module
class Recognizer2D(BaseRecognizer):
    """class for recognizer2d"""
    def __init__(self,
                 modality='RGB',
                 backbone='BNInception',
                 cls_head='TSNClsHead',
                 fcn_testing=False,
                 module_cfg=None,
                 nonlocal_cfg=None,
                 train_cfg=None,
                 test_cfg=None):
        super(Recognizer2D, self).__init__(backbone, cls_head)
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

    def _prepare_base_model(self, backbone, module_cfg, nonlocal_cfg):
        # module_cfg, example:
        # tsm: dict(type='tsm', n_frames=8 , n_div=8,
        #           shift_place='blockres',
        #           temporal_pool=False, two_path=False)
        # nolocal: dict(n_segment=8)
        backbone_name = backbone['type']
        module_name = module_cfg.pop('type')
        self.module_name = module_name
        if backbone_name == 'ResNet':
            # Add module for 2D backbone
            if module_name == 'MVF':
                print('Adding MVF module...')
                from ..modules.MVF import make_multi_view_fusion
                make_multi_view_fusion(self.backbone, **module_cfg)

            if module_name == 'CoST':
                print('Adding CoST...')
                from ..modules.CoST import make_CoST
                make_CoST(self.backbone, **module_cfg)

            if nonlocal_cfg:
                print('Adding non-local module...')
                from ..modules.local_attention import make_non_local
                make_non_local(self.backbone, **nonlocal_cfg)

        elif backbone_name == 'MobileNetV2':
            if module_name == 'tsm':
                from ..modules import TemporalShift
                from ..backbones import InvertedResidual
                for m in self.backbone.modules():
                    if isinstance(m, InvertedResidual) and len(
                            m.conv) == 8 and m.identity:
                        print('Adding temporal shift... {}'.format(
                            m.identity))
                        m.conv[0] = TemporalShift(
                            m.conv[0],
                            n_segment=module_cfg['n_segment'],
                            n_div=module_cfg['n_div'])

            elif module_name == 'MVF':
                print('Adding MVF module...')
                from ..backbones import InvertedResidual
                from ..modules.MVF import MVF
                for m in self.backbone.modules():
                    if isinstance(m, InvertedResidual) and len(
                            m.conv) == 8 and m.identity:
                        print('Adding adaptive fusion... {}'.format(
                            m.identity))
                        m.conv[0] = MVF(
                            m.conv[0],
                            n_segment=module_cfg['n_segment'],
                            in_channels=m.conv[0].in_channels,
                            alpha=module_cfg['alpha'],
                            share=module_cfg['share'],
                            mode=module_cfg['mode'])

    # TODO: Debug and test flow
    def _construct_2d_backbone_conv1(self, in_channels):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.backbone.modules())
        first_conv_idx = list(filter(lambda x: isinstance(
            modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, \
        # assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (in_channels, ) + kernel_size[2:]
        new_kernel_data = params[0].data.mean(dim=1, keepdim=True).expand(
            new_kernel_size).contiguous()  # make contiguous!

        new_conv_layer = nn.Conv2d(in_channels, conv_layer.out_channels,
                                   conv_layer.kernel_size, conv_layer.stride,
                                   conv_layer.padding,
                                   bias=True if len(params) == 2 else False)
        new_conv_layer.weight.data = new_kernel_data
        if len(params) == 2:
            new_conv_layer.bias.data = params[1].data
        # remove ".weight" suffix to get the layer layer_name
        layer_name = list(container.state_dict().keys())[0][:-7]
        setattr(container, layer_name, new_conv_layer)

    def forward_train(self, imgs, labels, **kwargs):
        """train"""
        #  [B S C H W]
        #  [BS C H W]
        num_batch = imgs.shape[0]
        imgs = imgs.reshape((-1, self.in_channels) + imgs.shape[3:])
        num_seg = imgs.shape[0] // num_batch

        x = self.extract_feat(imgs)  # 64 2048 7 7
        losses = dict()
        if self.with_cls_head:
            temporal_pool = imgs.shape[0] // x.shape[0]
            cls_score = self.cls_head(x, num_seg // temporal_pool)
            gt_label = labels.squeeze()
            loss_cls = self.cls_head.loss(cls_score, gt_label)
            losses.update(loss_cls)

        return losses

    def forward_test(self, imgs, return_numpy, **kwargs):
        """test"""
        #  imgs: [B tem*crop*clip C H W]
        #  imgs: [B*tem*crop*clip C H W]
        num_batch = imgs.shape[0]
        imgs = imgs.reshape((-1, self.in_channels) + imgs.shape[3:])
        num_frames = imgs.shape[0] // num_batch
        x = self.extract_feat(imgs)
        if self.with_cls_head:
            temporal_pool = imgs.shape[0] // x.shape[0]
            if self.module_cfg:
                if self.fcn_testing:
                    # view to 3D, [120, 2048, 8, 8] -> [30, 4, 2048, 8, 8]
                    x = x.reshape(
                        (-1, self.module_cfg['n_segment']//temporal_pool) + x.shape[1:])
                    x = x.transpose(1, 2)  # [30, 2048, 4, 8, 8]
                    cls_score = self.cls_head(
                        x, self.module_cfg['n_segment']//temporal_pool)  # [30 400]
                else:
                    # [120 2048 8 8] ->  [30 400]
                    cls_score = self.cls_head(
                        x, self.module_cfg['n_segment']//temporal_pool)
            else:
                cls_score = self.cls_head(x, num_frames // temporal_pool)
            cls_score = self.average_clip(cls_score)
        if return_numpy:
            return cls_score.cpu().numpy()
        else:
            return cls_score
