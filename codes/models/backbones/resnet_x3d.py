"""unofficial x3d implementation
Note: The code is only used to count #FLOPs of x3d.
"""
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import constant_init, kaiming_init
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.utils import _triple

from ...utils import get_root_logger, load_checkpoint
from ..builder import BACKBONES
from ..common import (SE3DModule, build_norm_layer, get_norm_type, HardSwish,
                      rgetattr, rhasattr)
from ..modules.local_attention import LocalAttention as NonLocalModule
from ..modules.local_attention import build_nonlocal_block
from .resnet import ResNet


def conv3x3x3(in_planes, out_planes, spatial_stride=1,
              temporal_stride=1, dilation=1):
    "3x3x3 convolution with padding"
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=(temporal_stride, spatial_stride, spatial_stride),
        padding=dilation,
        dilation=dilation,
        bias=False)


def conv1x3x3(in_planes, out_planes, spatial_stride=1,
              temporal_stride=1, dilation=1):
    "1x3x3 convolution with padding"
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(1, 3, 3),
        stride=(temporal_stride, spatial_stride, spatial_stride),
        padding=(0, dilation, dilation),
        dilation=dilation,
        bias=False)


class BasicBlock(nn.Module):
    """basic block"""
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 spatial_stride=1,
                 temporal_stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 if_inflate=True,
                 inflate_style=None,
                 norm_cfg=dict(type='BN3d'),
                 with_cp=False):
        super(BasicBlock, self).__init__()
        if if_inflate:
            self.conv1 = conv3x3x3(inplanes, planes, spatial_stride,
                                   temporal_stride, dilation)
        else:
            self.conv1 = conv1x3x3(inplanes, planes, spatial_stride,
                                   temporal_stride, dilation)
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.add_module(self.norm1_name, norm1)
        self.add_module(self.norm2_name, norm2)
        # self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        if if_inflate:
            self.conv2 = conv3x3x3(planes, planes)
        else:
            self.conv2 = conv1x3x3(planes, planes)
        # self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.spatial_stride = spatial_stride
        self.temporal_stride = temporal_stride
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
    """Bottleneck block for ResNet3D.

    Args:
        inplanes (int): Number of channels for the input in first conv3d layer.
        planes (int): Number of channels produced by some norm/conv3d layers.
        spatial_stride (int): Spatial stride in the conv3d layer, Default 1.
        temporal_stride (int): Temporal stride in the conv3d layer, Default 1.
        dilation (int): Spacing between kernel elements, Default 1.
        downsample (obj): Downsample layer. Default None.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer. Default 'pytorch'.
        inflate (bool): Whether to inflate kernel, Default True.
        inflate_style (str): `3x1x1` or `1x1x1`. which determines the kernel
            sizes and padding strides for conv1 and conv2 in each block.
            Default '3x1x1'.
        norm_cfg (dict): Config for norm layers. required keys are `type`,
            Default dict(type='BN3d').
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed, Default False.
    """

    def __init__(self,
                 inplanes,
                 planes,
                 spatial_stride=1,
                 temporal_stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 if_inflate=True,
                 inflate_style='3x1x1',
                 norm_cfg=dict(type='BN3d'),
                 if_nonlocal=True,
                 nonlocal_cfg=None,
                 with_cp=False,
                 ratio_bottleneck=2.25,
                 with_se=True,
                 with_hs=True):

        super(Bottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']
        assert inflate_style in ['3x1x1', '3x3x3', 'mobile']
        self.inplanes = inplanes
        self.planes = planes
        self.downsample = downsample
        self.spatial_tride = spatial_stride
        self.temporal_tride = temporal_stride
        self.dilation = dilation
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.with_se = with_se
        norm_type = get_norm_type(norm_cfg)

        if style == 'pytorch':
            self.conv1_stride_s = 1
            self.conv2_stride_s = spatial_stride
            self.conv1_stride_t = 1
            self.conv2_stride_t = temporal_stride
        else:
            self.conv1_stride_s = spatial_stride
            self.conv2_stride_s = 1
            self.conv1_stride_t = temporal_stride
            self.conv2_stride_t = 1

        if if_inflate:
            if inflate_style == '3x1x1':
                conv1_kernel_size = (3, 1, 1)
                conv1_padding = (1, 0, 0)
                conv2_kernel_size = (1, 3, 3)
                conv2_padding = (0, dilation, dilation)
            else:
                conv1_kernel_size = (1, 1, 1)
                conv1_padding = 0
                conv2_kernel_size = (3, 3, 3)
                conv2_padding = (1, dilation, dilation)

            self.conv1 = nn.Conv3d(
                inplanes,
                planes,
                kernel_size=conv1_kernel_size,
                stride=(self.conv1_stride_t, self.conv1_stride_s,
                        self.conv1_stride_s),
                padding=conv1_padding,
                bias=False)
            self.conv2 = nn.Conv3d(
                planes,
                planes,
                kernel_size=conv2_kernel_size,
                stride=(self.conv2_stride_t, self.conv2_stride_s,
                        self.conv2_stride_s),
                padding=conv2_padding,
                dilation=(1, dilation, dilation),
                bias=False,
                groups=planes if inflate_style == 'mobile' else 1)

        else:
            self.conv1 = nn.Conv3d(
                inplanes,
                planes,
                kernel_size=1,
                stride=(1, self.conv1_stride_s, self.conv1_stride_s),
                bias=False)
            self.conv2 = nn.Conv3d(
                planes,
                planes,
                kernel_size=(1, 3, 3),
                stride=(1, self.conv2_stride_s, self.conv2_stride_s),
                padding=(0, dilation, dilation),
                dilation=(1, dilation, dilation),
                bias=False)

        self.conv3 = nn.Conv3d(
            planes, int(planes / ratio_bottleneck), kernel_size=1, bias=False)

        self.activation = HardSwish() if with_hs else nn.ReLU(inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.se = SE3DModule(planes, 16, with_hs) if with_se else None
        self.bn1 = norm_type(planes)
        self.bn2 = norm_type(planes)

        self.bn3 = norm_type(int(planes / ratio_bottleneck))
        if if_nonlocal and nonlocal_cfg is not None:
            nonlocal_cfg_ = nonlocal_cfg.copy()
            nonlocal_cfg_['in_channels'] = int(planes / ratio_bottleneck)
            self.nonlocal_block = build_nonlocal_block(nonlocal_cfg_)
        else:
            self.nonlocal_block = None

    def forward(self, x):
        """forward"""
        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.activation(out)

            out = self.conv2(out)
            out = self.bn2(out)
            if self.se is not None:
                out = self.se(out)
            out = self.activation(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        if self.nonlocal_block is not None:
            out = self.nonlocal_block(out)

        return out


def make_res_layer(block,
                   inplanes,
                   planes,
                   blocks,
                   spatial_stride=1,
                   temporal_stride=1,
                   dilation=1,
                   style='pytorch',
                   inflate_freq=1,
                   inflate_style='3x1x1',
                   norm_cfg=None,
                   nonlocal_freq=1,
                   nonlocal_cfg=None,
                   with_cp=False,
                   ratio_bottleneck=2.25):
    """Build residual layer for ResNet3D.

    Args:
        block (nn.Module): Residual module to be built.
        inplanes (int): Number of channels for the input feature in each block.
        planes (int): Number of channels for the output feature in each block.
        blocks (int): Number of residual blocks.
        spatial_stride (int | Sequence[int]): Spatial strides in residual and
            conv layers, Default 1.
        temporal_stride (int | Sequence[int]): Temporal strides in residual and
            conv layers, Default 1.
        dilation (int): Spacing between kernel elements, Default 1.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer. Default 'pytorch'.
        inflate (int | Sequence[int]): Determine whether to inflate for each
            block, Default 1.
        inflate_style (str): `3x1x1` or `1x1x1`. which determines the kernel
            sizes and padding strides for conv1 and conv2 in each block.
            Default '3x1x1'.
        norm_cfg (dict): Config for norm layers, Default None.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed, Default False.

    Returns:
        A residual layer for the given config.
    """
    inflate_freq = inflate_freq if not isinstance(
        inflate_freq, int) else (inflate_freq, ) * blocks
    nonlocal_freq = nonlocal_freq if not isinstance(
        nonlocal_freq, int) else (nonlocal_freq, ) * blocks
    assert len(inflate_freq) == blocks
    assert len(nonlocal_freq) == blocks
    downsample = None
    if spatial_stride != 1 or inplanes != int(planes / ratio_bottleneck):
        norm_type = get_norm_type(norm_cfg)
        downsample = nn.Sequential(
            nn.Conv3d(
                inplanes,
                int(planes / ratio_bottleneck),
                kernel_size=1,
                stride=(temporal_stride, spatial_stride, spatial_stride),
                bias=False),
            norm_type(int(planes / ratio_bottleneck)),
        )

    layers = []
    layers.append(
        block(
            inplanes,
            planes,
            spatial_stride,
            temporal_stride,
            dilation,
            downsample,
            style=style,
            if_inflate=(inflate_freq[0] == 1),
            inflate_style=inflate_style,
            norm_cfg=norm_cfg,
            if_nonlocal=(nonlocal_freq[0] == 1),
            nonlocal_cfg=nonlocal_cfg,
            with_cp=with_cp,
            ratio_bottleneck=ratio_bottleneck))
    inplanes = int(planes / ratio_bottleneck)
    for i in range(1, blocks):
        layers.append(
            block(inplanes,
                  planes,
                  1, 1,
                  dilation,
                  style=style,
                  if_inflate=(inflate_freq[i] == 1),
                  inflate_style=inflate_style,
                  norm_cfg=norm_cfg,
                  if_nonlocal=(nonlocal_freq[i] == 1),
                  nonlocal_cfg=nonlocal_cfg,
                  with_cp=with_cp,
                  ratio_bottleneck=ratio_bottleneck))

    return nn.Sequential(*layers)


@BACKBONES.register_module
class ResNet_X3D(nn.Module):
    """ResNet_X3D backbone.
    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        pretrained (str): Name of pretrained model.
        pretrained2d (bool): Whether to load pretrained 2D model, Default True.
        in_channels (int): Channel num of input features, Default 3.
        num_stages (int): Resnet stages, Default 4.
        spatial_strides (Sequence[int]):
            Spatial strides of residual blocks of each stage.
        temporal_strides (Sequence[int]):
            Temporal strides of residual blocks of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        conv1_kernel (Sequence[int]): Kernel size of the first conv layer.
        conv1_stride_t (int): Temporal stride of the first conv layer.
        pool1_stride_t (int): Temporal stride of the first pooling layer.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        inflate (Sequence[int]): Inflate Dims of each block.
        inflate_stride (Sequence[int]):
            Inflate stride of each block.
        inflate_style (str): `3x1x1` or `1x1x1`. which determines the kernel
            sizes and padding strides for conv1 and conv2 in each block.
        norm_cfg (dict): Config for norm layers. required keys are `type` and
            `requires_grad`, Default dict(type='BN3d', requires_grad=True).
        norm_eval (bool): Whether to set BN layers to eval mode, namely, freeze
            running stats (mean and var), Default True.
        norm_frozen (bool): Whether to freeze weight and bias of BN layers.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed, Default False.
        zero_init_residual (bool):
            Whether to use zero initialization for residual block,
            Default True.
    """

    arch_settings = {
        1: (Bottleneck, (1, 2, 5, 3)),
        2.2: (Bottleneck, (3, 5, 11, 7)),
        5: (Bottleneck, (5, 10, 25, 15)),
        # 101: (Bottleneck, (3, 4, 23, 3)),
        # 152: (Bottleneck, (3, 8, 36, 3)),
        # 200: (Bottleneck, (3, 24, 36, 3)),
    }

    def __init__(self,
                 depth,
                 pretrained=None,
                 pretrained2d=True,
                 in_channels=3,
                 num_stages=4,
                 spatial_strides=(1, 2, 2, 2),
                 temporal_strides=(1, 1, 1, 1),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 conv1_kernel=(5, 7, 7),
                 ratio_width=1,
                 ratio_up=2,
                 ratio_bottleneck=2.25,
                 ratio_depth=2.2,
                 conv1_stride_t=2,
                 pool1_kernel_t=1,
                 pool1_stride_t=2,
                 style='pytorch',
                 frozen_stages=-1,
                 # For C2D baseline, inflate_freq set to -1.
                 inflate_freq=(1, 1, 1, 1),
                 inflate_stride=(1, 1, 1, 1),
                 inflate_style='3x1x1',
                 norm_cfg=dict(type='BN3d', requires_grad=True),
                 nonlocal_stages=(-1, ),
                 nonlocal_freq=(0, 1, 1, 0),
                 nonlocal_cfg=None,
                 no_pool2=False,
                 norm_eval=True,
                 norm_frozen=False,
                 partial_norm=False,
                 with_cp=False,
                 zero_init_residual=True):
        super(ResNet_X3D, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        self.depth = depth
        self.in_channels = in_channels
        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.spatial_strides = spatial_strides
        self.temporal_strides = temporal_strides
        self.dilations = dilations
        assert len(spatial_strides) == len(
            temporal_strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.frozen_stages = frozen_stages
        self.inflate_freqs = inflate_freq if not isinstance(
            inflate_freq, int) else (inflate_freq, ) * num_stages
        self.inflate_style = inflate_style
        self.nonlocal_stages = nonlocal_stages
        self.nonlocal_freqs = nonlocal_freq if not isinstance(
            nonlocal_freq, int) else (nonlocal_freq, ) * num_stages
        self.nonlocal_cfg = nonlocal_cfg
        self.norm_eval = norm_eval
        self.norm_frozen = norm_frozen
        self.partial_norm = partial_norm
        self.with_cp = with_cp
        self.norm_cfg = norm_cfg
        self.zero_init_residual = zero_init_residual

        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]

        input_channel = int(24 * ratio_width)
        self.conv1 = nn.Conv3d(
            in_channels, input_channel,
            kernel_size=conv1_kernel,
            stride=(conv1_stride_t, 2, 2),
            padding=tuple([(k - 1) // 2 for k in _triple(conv1_kernel)]),
            bias=False)
        norm_type = get_norm_type(norm_cfg)
        self.conv1_3x1 = nn.Sequential(
            nn.Conv3d(
                input_channel, input_channel,
                kernel_size=(3, 1, 1),
                stride=1,
                padding=(1, 0, 0),
                dilation=1,
                groups=input_channel,
                bias=False),
            norm_type(input_channel),
            nn.ReLU(inplace=True),
            # pw-linear
            # nn.Conv3d(input_channel, input_channel, 1, 1, 0, bias=False),
        )

        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, 3, postfix=1)
        self.add_module(self.norm1_name, norm1)
        # self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(
            kernel_size=(pool1_kernel_t, 3, 3),
            stride=(pool1_stride_t, 2, 2),
            padding=(pool1_kernel_t // 2, 1, 1))
        # TODO: Check whether pad=0 differs a lot
        self.pool2 = nn.MaxPool3d(
            kernel_size=(2, 1, 1),
            stride=(2, 1, 1),
            padding=(0, 0, 0))
        self.no_pool2 = no_pool2

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            spatial_stride = spatial_strides[i]
            temporal_stride = temporal_strides[i]
            dilation = dilations[i]
            inplanes = int(24 * ratio_width * ratio_up **
                           (i - 1)) if i > 0 else int(24 * ratio_width)
            planes = int(24 * ratio_width * ratio_bottleneck * 2 ** i)
            res_layer = make_res_layer(
                self.block,
                inplanes,
                planes,
                num_blocks,
                spatial_stride=spatial_stride,
                temporal_stride=temporal_stride,
                dilation=dilation,
                style=self.style,
                inflate_freq=self.inflate_freqs[i],
                inflate_style=self.inflate_style,
                norm_cfg=norm_cfg,
                nonlocal_freq=self.nonlocal_freqs[i],
                nonlocal_cfg=nonlocal_cfg if i in nonlocal_stages else None,
                with_cp=with_cp,
                ratio_bottleneck=ratio_bottleneck)
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        feat_dim = int(24 * ratio_width * 2 ** (
            len(self.stage_blocks) - 1))
        self.conv5 = nn.Conv3d(
            feat_dim,
            int(feat_dim * ratio_bottleneck),
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.pool5 = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Conv3d(
            int(feat_dim * ratio_bottleneck),
            2048,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)

    @property
    def norm1(self):
        """norm1"""
        return getattr(self, self.norm1_name)

    def init_weights(self):
        """init weight"""
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info("load model from: {}".format(self.pretrained))
            if self.pretrained2d:
                resnet2d = ResNet(self.depth)
                load_checkpoint(resnet2d, self.pretrained,
                                strict=False, logger=logger)
                for name, module in self.named_modules():
                    if isinstance(module, NonLocalModule):
                        module.init_weights()
                    elif isinstance(module, nn.Conv3d) and rhasattr(
                            resnet2d, name):
                        new_weight = rgetattr(
                            resnet2d, name).weight.data.unsqueeze(2).expand_as(
                                module.weight) / module.weight.data.shape[2]
                        module.weight.data.copy_(new_weight)
                        logger.info(
                            "{}.weight loaded from weights file into {}".
                            format(name, new_weight.shape))

                        if hasattr(module, 'bias') and module.bias is not None:
                            new_bias = rgetattr(resnet2d, name).bias.data
                            module.bias.data.copy_(new_bias)
                            logger.info(
                                "{}.bias loaded from weights file into {}".
                                format(name, new_bias.shape))

                    elif isinstance(module, _BatchNorm) and rhasattr(
                            resnet2d, name):
                        for attr in [
                                'weight', 'bias', 'running_mean', 'running_var'
                        ]:
                            logger.info(
                                "{}.{} loaded from weights file into {}"
                                .format(
                                    name, attr, getattr(
                                        rgetattr(resnet2d, name), attr).shape))
                            setattr(module, attr, getattr(
                                rgetattr(resnet2d, name), attr))
            else:
                load_checkpoint(
                    self, self.pretrained, strict=False, logger=logger)

        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    kaiming_init(m)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)
            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """forward"""
        print(x.shape)
        x = self.conv1(x)
        # x = self.norm1(x)
        x = self.relu(x)
        print(x.shape)
        # x = self.maxpool(x)
        x = self.conv1_3x1(x)
        print(x.shape)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
            if self.no_pool2:
                pass
            else:
                if i == 0:
                    x = self.pool2(x)
            print(x.shape)
        # if len(outs) == 1:
        #     return outs[0]
        # else:
        #     return tuple(outs)
        x = self.conv5(x)
        print(x.shape)
        x = self.pool5(x)
        print(x.shape)
        x = self.fc1(x)
        print(x.shape)
        return x

    def train(self, mode=True):
        """train"""
        super(ResNet_X3D, self).train(mode)
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
            for param in self.bn1.parameters():
                param.requires_grad = False
            self.norm1.eval()
            self.norm1.weight.requires_grad = False
            self.norm1.bias.requires_grad = False
            for i in range(1, self.frozen_stages + 1):
                mod = getattr(self, 'layer{}'.format(i))
                mod.eval()
                for param in mod.parameters():
                    param.requires_grad = False
