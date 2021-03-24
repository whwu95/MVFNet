"""resnet"""
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import constant_init, kaiming_init
from torch.nn.modules.batchnorm import _BatchNorm

from ...utils import get_root_logger, load_checkpoint
from ..builder import BACKBONES
from ..common import build_norm_layer, get_norm_type


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False)


class BasicBlock(nn.Module):
    """Basic block for ResNet.

    Args:
        inplanes (int): Number of channels for the input in first conv2d layer.
        planes (int): Number of channels produced by some norm/conv2d layers.
        stride (int): Stride in the conv layer. Default 1.
        dilation (int): Spacing between kernel elements. Default 1.
        downsample (obj): Downsample layer. Default None.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer. Default 'pytorch'.
        norm_cfg (dict): Config for norm layers. required keys are `type`,
            Default dict(type='BN').
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default False.
    """
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 norm_cfg=dict(type='BN'),
                 with_cp=False):
        super(BasicBlock, self).__init__()
        assert style in ['pytorch', 'caffe']
        self.conv1 = conv3x3(inplanes, planes, stride, dilation)

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.add_module(self.norm1_name, norm1)
        # self.bn1 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)

        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.add_module(self.norm2_name, norm2)
        # self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.style = style
        self.stride = stride
        self.dilation = dilation
        self.norm_cfg = norm_cfg
        assert not with_cp

    @property
    def norm1(self):
        """norm1"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """norm2"""
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """forward"""
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck block for ResNet.

    Args:
        inplanes (int):
            Number of channels for the input feature in first conv layer.
        planes (int):
            Number of channels produced by some norm layes and conv layers
        stride (int): Spatial stride in the conv layer. Default 1.
        dilation (int): Spacing between kernel elements. Default 1.
        downsample (obj): Downsample layer. Default None.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer. Default 'pytorch'.
        norm_cfg (dict): Config for norm layers. required keys are `type`,
            Default dict(type='BN').
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default False.
    """
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 norm_cfg=dict(type='BN'),
                 with_cp=False,
                 avd=False,
                 avd_first=False):
        """Bottleneck block for ResNet.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']
        self.inplanes = inplanes
        self.planes = planes
        self.avd = avd and stride > 1
        self.avd_first = avd_first

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        if style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=self.conv2_stride,
            padding=dilation,
            dilation=dilation,
            bias=False)

        # self.bn1 = nn.BatchNorm2d(planes)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.add_module(self.norm1_name, norm1)
        self.add_module(self.norm2_name, norm2)

        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False)

        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)
        self.add_module(self.norm3_name, norm3)
        # self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp

    @property
    def norm1(self):
        """norm1"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """norm2"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """norm3"""
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """forward"""
        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if self.avd and self.avd_first:
                out = self.avd_layer(out)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            if self.avd and not self.avd_first:
                out = self.avd_layer(out)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


def make_res_layer(block,
                   inplanes,
                   planes,
                   blocks,
                   stride=1,
                   dilation=1,
                   style='pytorch',
                   norm_cfg=None,
                   with_cp=False,
                   avg_down=False,
                   avd=False,
                   avd_first=False):
    """Build residual layer for ResNet.

    Args:
        block: (nn.Module): Residual module to be built.
        inplanes (int): Number of channels for the input feature in each block.
        planes (int): Number of channels for the output feature in each block.
        blocks (int): Number of residual blocks.
        stride (int): Stride in the conv layer. Default 1.
        dilation (int): Spacing between kernel elements. Default 1.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer. Default 'pytorch'.
        norm_cfg (dict): Config for norm layers. required keys are `type`,
            Default None.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default False.

    Returns:
        A residual layer for the given config.
    """
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        norm_type = get_norm_type(norm_cfg)
        down_layers = []
        if avg_down:
            if dilation == 1:
                down_layers.append(
                    nn.AvgPool2d(kernel_size=stride,
                                 stride=stride,
                                 ceil_mode=True,
                                 count_include_pad=False))
            else:
                down_layers.append(
                    nn.AvgPool2d(kernel_size=1,
                                 stride=1,
                                 ceil_mode=True,
                                 count_include_pad=False))
            down_layers.append(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False))
        else:
            down_layers.append(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False))
        down_layers.append(norm_type(planes * block.expansion))
        downsample = nn.Sequential(*down_layers)

    layers = []
    layers.append(
        block(
            inplanes,
            planes,
            stride,
            dilation,
            downsample,
            style=style,
            norm_cfg=norm_cfg,
            with_cp=with_cp,
            avd=avd,
            avd_first=avd_first))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(
            block(inplanes, planes, 1, dilation,
                  style=style, norm_cfg=norm_cfg, with_cp=with_cp,
                  avd=avd, avd_first=avd_first))

    return nn.Sequential(*layers)


@BACKBONES.register_module
class ResNet(nn.Module):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        pretrained (str): Name of pretrained model. Default None.
        num_stages (int): Resnet stages. Default 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer. Default `pytorch`.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters. Default -1.
        norm_cfg (dict):
            Config for norm layers. required keys are `type` and
            `requires_grad`. Default dict(type='BN', requires_grad=True).
        norm_eval (bool): Whether to set BN layers to eval mode, namely, freeze
            running stats (mean and var). Default True.
        bn_frozen (bool): Whether to freeze weight and bias of BN layersn
            Default False.
        partial_bn (bool): Whether to freeze weight and bias of **all
            but the first** BN layersn Default False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default False.
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 pretrained=None,
                 in_channels=3,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 frozen_stages=-1,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 norm_frozen=False,
                 partial_norm=False,
                 with_cp=False,
                 avg_down=False,
                 avd=False,
                 avd_first=False,
                 deep_stem=False,
                 stem_width=64):
        super(ResNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        self.depth = depth
        self.in_channels = in_channels
        self.pretrained = pretrained
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.frozen_stages = frozen_stages
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.norm_frozen = norm_frozen
        self.partial_norm = partial_norm
        self.with_cp = with_cp

        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_width * 2 if deep_stem else 64
        norm_type = get_norm_type(norm_cfg)
        if deep_stem:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, stem_width, kernel_size=3, stride=2,
                          padding=1, bias=False),
                norm_type(stem_width),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_width, stem_width, kernel_size=3,
                          stride=1, padding=1, bias=False),
                norm_type(stem_width),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_width, stem_width * 2, kernel_size=3,
                          stride=1, padding=1, bias=False),
            )
        else:
            self.conv1 = nn.Conv2d(
                3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, self.inplanes, postfix=1)
        self.add_module(self.norm1_name, norm1)
        # self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            planes = 64 * 2 ** i
            res_layer = make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                avg_down=avg_down,
                avd=avd,
                avd_first=avd_first)
            self.inplanes = planes * self.block.expansion
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.feat_dim = self.block.expansion * 64 * 2 ** (
            len(self.stage_blocks) - 1)

    @property
    def norm1(self):
        """norm1"""
        return getattr(self, self.norm1_name)

    def init_weights(self):
        """init weight"""
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, self.pretrained,
                            map_location='cpu', strict=False, logger=logger)
        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """forward"""
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def train(self, mode=True):
        """train"""
        super(ResNet, self).train(mode)
        if self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
                    if self.norm_frozen:
                        for params in m.parameters():
                            params.requires_grad = False
        if self.partial_norm:
            for i in range(1, self.frozen_stages + 1):
                mod = getattr(self, 'layer{}'.format(i))
                for m in mod.modules():
                    if isinstance(m, _BatchNorm):
                        m.eval()
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

        if mode and self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
            for param in self.norm1.parameters():
                param.requires_grad = False
            self.norm1.eval()
            self.norm1.weight.requires_grad = False
            self.norm1.bias.requires_grad = False
            for i in range(1, self.frozen_stages + 1):
                mod = getattr(self, 'layer{}'.format(i))
                mod.eval()
                for param in mod.parameters():
                    param.requires_grad = False
