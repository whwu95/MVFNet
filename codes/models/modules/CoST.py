"""unofficial Cost implementation without self-attention
Note: The code is only used to count #FLOPs of CoST for MVFNet paper.
R50_8x8: 45.8G
R101_8x8: 107G
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core import auto_fp16
from ..common import HardSwish



def make_CoST(
        net, n_segment, place='blockres',
        temporal_pool=False, two_path=False, shift_freq=(1, 1, 1, 1)):
    """make CoST"""
    n_segment_list = [n_segment] * 4
    assert n_segment_list[-1] > 0
    print('=> n_segment per stage: {}'.format(n_segment_list))

    # import torchvision
    if True:  # isinstance(net, torchvision.models.ResNet):
        if 'blockres' in place:
            n_round = 1
            if len(list(net.layer3.children())) >= 23:
                # n_round = 2
                print('=> Using n_round {} to insert temporal shift'.
                      format(n_round))

            def make_block_temporal(stage, this_segment):
                """Make CoST block"""
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks residual'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    if i != 0:
                        blocks[i].conv2 = SimpleCoST(
                            b.conv2, this_segment, blocks[i].conv2.in_channels)
                return nn.Sequential(*blocks)

            net.layer1 = make_block_temporal(
                net.layer1, n_segment_list[0]) if shift_freq[0] else net.layer1
            net.layer2 = make_block_temporal(
                net.layer2, n_segment_list[1]) if shift_freq[1] else net.layer2
            net.layer3 = make_block_temporal(
                net.layer3, n_segment_list[2]) if shift_freq[2] else net.layer3
            net.layer4 = make_block_temporal(
                net.layer4, n_segment_list[3]) if shift_freq[3] else net.layer4
    else:
        raise NotImplementedError(place)




class SimpleCoST(nn.Module):
    """Simple CoST """
    def __init__(self, net, n_segment, in_channels):
        super(SimpleCoST, self).__init__()
        # self.net = net
        self.n_segment = n_segment

        self.bn = nn.BatchNorm3d(in_channels)
        self.activation = nn.ReLU(inplace=True)
        self.shift_conv = nn.Conv3d(
            in_channels, in_channels, [1, 3, 3], stride=1, padding=[0, 1, 1], bias=False)                                                           
        self._initialize_weights()
        print('=> Using CoST...')

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # kaiming_init(m)
                n = m.kernel_size[0] * m.kernel_size[1] * \
                    m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        """forward"""
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w).transpose(1, 2)  # n, c, t, h, w
        # get H & W
        
        tmp_t = self.shift_conv(x)
        tmp_h = self.shift_conv(x.transpose(2, 3)).transpose(2, 3)
        tmp_w = self.shift_conv(x.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)

        x = tmp_t + tmp_h + tmp_w  # n, c, t, h, w

        # add bn and activation
        x = self.bn(x)
        x = self.activation(x)
        x = x.transpose(1, 2).contiguous().view(nt, c, h, w)
        return x
        # return self.net(x)
