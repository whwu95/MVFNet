"""Nonlocal Module"""
from torch import nn, Tensor
import torch
import torch.nn.functional as F
# from callonce import printonce
import logging
from torch.jit import Final

_logger = logging.getLogger(__name__)


class LocalAttention(nn.Module):
    # _use_time_shift: Final[bool]
    # _use_time_weighting: Final[bool]
    # _dim: Final[int]
    # _hidden: Final[int]
    # _kernel_size: Final[int]
    # _k2: Final[int]
    # _padding: Final[int]
    # _instantiation: Final[str]

    def __init__(
        self,
        dim: int,
        hidden: int,
        kernel_size: int = 3,
        padding: int = 1,
        instantiation='dot_product',
        use_time_shift: bool = False,
        time_weighting_size=None,
    ) -> None:
        super().__init__()

        # self.conv_theta = nn.Conv3d(
        #     dim, hidden, 1, stride=1, padding=0,
        # )
        # self.conv_phi = nn.Conv3d(
        #     dim, hidden, 1, stride=1, padding=0,
        # )
        # self.conv_g = nn.Conv3d(
        #     dim, hidden, 1, stride=1, padding=0,
        # )
        self.conv_in = nn.Conv3d(
            dim, hidden * 3, 1, stride=1, padding=0,
        )
        self.conv_out = nn.Conv3d(
            hidden, dim, 1, stride=1, padding=0,
        )
        if use_time_shift:
            self.time_shift = nn.ZeroPad2d((0, 0, 0, 0, 1, 0))
            _logger.info('time shift padding: %s', self.time_shift)
        else:
            self.time_shift = None

        use_time_weighting = time_weighting_size is not None
        if use_time_weighting:
            self.time_weighting = nn.Parameter(
                torch.ones(time_weighting_size)
            )
            _logger.info('time weighing: %s', self.time_weighting.shape)
        else:
            self.time_weighting = None

        self.bn = nn.BatchNorm3d(dim)
        self.unfold = nn.Unfold(kernel_size, padding=padding)
        self._dim = dim
        self._hidden = hidden
        self._kernel_size = kernel_size
        self._k2 = kernel_size * kernel_size
        self._padding = padding
        self._instantiation = instantiation
        self._use_time_shift = use_time_shift
        self._use_time_weighting = use_time_weighting

        _logger.info(
            f'init {self.__class__.__qualname__}(dim: {self._dim}, hidden: {self._hidden})'
        )

    def forward(self, x: Tensor) -> Tensor:
        x_identity = x
        B, C, T, H, W = x.shape
        hidden = self.conv_in(x)
        theta, phi, g = hidden.chunk(3, dim=1)

        theta_unf = self._unfold_and_view(theta)
        phi_unf = self._unfold_and_view(phi)
        g_unf = self._unfold_and_view(g)

        theta_phi: Tensor = torch.einsum('ncts,ncps->ntps', theta_unf, phi_unf)

        if self._instantiation == 'softmax':
            # Normalizing the affinity tensor theta_phi before softmax.
            theta_phi = theta_phi * (self._hidden ** -0.5)
            theta_phi = F.softmax(theta_phi, dim=2)
        elif self._instantiation == 'dot_product':
            spatial_temporal_dim = theta_phi.shape[2]
            # printonce('st dim', spatial_temporal_dim)
            theta_phi = theta_phi / spatial_temporal_dim
        else:
            raise NotImplementedError(
                "Unknown norm type {}".format(self._instantiation)
            )

        if self._use_time_weighting:
            # https://github.com/BlinkDL/minGPT-tuned/blob/master/mingpt/model.py#L68
            theta_phi = theta_phi * self.time_weighting

        theta_phi_g: Tensor = torch.einsum('ntgs,ncgs->ncts', theta_phi, g_unf)

        # [B, C, T*K*K, H*W] -> [B, C*T*K*K, H*W]
        theta_phi_g = theta_phi_g.reshape(
            B, self._hidden * T * self._k2, H * W
        )

        # [B, T, C, K*K, H*W] -> [B*T, C*K*K, H*W]
        # theta_phi_g = theta_phi_g.transpose(
        #     1, 2).reshape(B * T, self._hidden * self._k2, H * W)
        out = F.fold(
            theta_phi_g, (H, W), self._kernel_size,
            padding=self._padding
        )
        # out = out.view(B, T, self._hidden, H, W).transpose(1, 2)
        out = out.view(B, self._hidden, T, H, W)
        out = self.conv_out(out)
        out = self.bn(out)
        return x_identity + out

    def _unfold_and_view(self, x: Tensor):
        B, C, T, H, W = x.shape

        if self._use_time_shift:
            x = self._time_shift(x)

        # [B, T, C, H, W]
        # x = x.transpose(1, 2)

        # x = x.reshape(B * T, C, H, W)
        x = x.view(B, C*T, H, W)

        # [B, C*T*K*K, H*W]
        x_unf = self.unfold(x)

        # printonce('x_unf', x_unf.shape)

        # [B, T, C, K*K, H*W]
        # x_unf = x_unf.reshape(B, T, C, self._k2, H * W)

        # [B, C, T, K*K, H*W]
        # x_unf = x_unf.transpose(1, 2)

        # [B, C, T*K*K, H*W]
        x_unf = x_unf.view(B, C, T * self._k2, H * W)

        return x_unf

    def _time_shift(self, x: Tensor):
        """
        https://github.com/BlinkDL/minGPT-tuned/blob/master/mingpt/model.py#L68
        """
        B, C, T, H, W = x.shape
        C_half = C // 2
        return torch.cat([
            self.time_shift(x)[:, :C_half, :T], x[:, C_half:]
        ], dim=1)

    def init_weights(self):
        pass


class NL3DWrapper(nn.Module):
    def __init__(self, block, n_segment):
        super(NL3DWrapper, self).__init__()
        self.block = block
        # self.nl = NonLocalModule(block.bn3.num_features, dim=3)
        self.nl = LocalAttention(
            block.bn3.num_features,
            block.bn3.num_features // 2,
        )
        self.n_segment = n_segment

    def forward(self, x):
        x = self.block(x)

        nt, c, h, w = x.size()
        x = x.view(nt // self.n_segment, self.n_segment, c,
                   h, w).transpose(1, 2)  # n, c, t, h, w
        x = self.nl(x)
        x = x.transpose(1, 2).contiguous().view(nt, c, h, w)
        return x


def make_non_local(net, n_segment):
    # import torchvision
    # import archs
    # isinstance(net, torchvision.models.ResNet) or \
    # isinstance(net, archs.small_resnet.ResNet):
    if True:
        len_layer2 = len(net.layer2)
        len_layer3 = len(net.layer3)
        nl_num_layer2 = 2
        nl_num_layer3 = 3
        nl_stride_layer2 = len_layer2 / nl_num_layer2  # 4/2=2
        # 6/3 for r50, 23/3 for r101
        nl_stride_layer3 = len_layer3 / nl_num_layer3

        layer2 = nn.Sequential()
        for idx in range(len_layer2):
            if idx % nl_stride_layer2 == nl_stride_layer2 - 1:
                layer2.add_module(str(idx), NL3DWrapper(
                    net.layer2[idx], n_segment))
            else:
                layer2.add_module(str(idx), net.layer2[idx])
        net.layer2 = layer2

        layer3 = nn.Sequential()
        for idx in range(len_layer3):
            if idx % nl_stride_layer3 == nl_stride_layer3 - 1:
                layer3.add_module(str(idx), NL3DWrapper(
                    net.layer3[idx], n_segment))
            else:
                layer3.add_module(str(idx), net.layer3[idx])
        net.layer3 = layer3

        # net.layer2 = nn.Sequential(
        #     NL3DWrapper(net.layer2[0], n_segment),
        #     net.layer2[1],
        #     NL3DWrapper(net.layer2[2], n_segment),
        #     net.layer2[3],
        # )
        # net.layer3 = nn.Sequential(
        #     NL3DWrapper(net.layer3[0], n_segment),
        #     net.layer3[1],
        #     NL3DWrapper(net.layer3[2], n_segment),
        #     net.layer3[3],
        #     NL3DWrapper(net.layer3[4], n_segment),
        #     net.layer3[5],
        # )
    else:
        raise NotImplementedError


def build_nonlocal_block(cfg):
    """ Build nonlocal block

    Args:
    """
    assert isinstance(cfg, dict)
    cfg_ = cfg.copy()

    dim = cfg_['in_channels']
    embed = cfg_.get('embed', True)
    embed_dim = cfg_.get('embed_dim', None)

    if embed_dim is None:
        embed_dim = dim // 2

    hidden = embed_dim

    la = LocalAttention(dim, hidden)

    # la = torch.jit.script(la)

    return la


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    la = LocalAttention(3, 8, instantiation='dot_product',
                        use_time_shift=True, time_weighting_size=(36, 36, 196))
    x = torch.rand(2, 3, 4, 14, 14)
    y = la(x)
    print(y.shape)
