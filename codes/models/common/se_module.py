"""SE Module (2D / 3D Version) """
import torch.nn as nn


class HardSigmoid(nn.Module):
    """h_sigmoid"""
    def __init__(self, inplace=True):
        super(HardSigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        """forward"""
        return self.relu(x + 3) / 6


class HardSwish(nn.Module):
    """h_swish"""
    def __init__(self, inplace=True):
        super(HardSwish, self).__init__()
        self.sigmoid = HardSigmoid(inplace=inplace)

    def forward(self, x):
        """forward"""
        return x * self.sigmoid(x)


class SE3DModule(nn.Module):
    """SE 3D"""
    def __init__(self, channels, reduction, use_hs=False):
        super(SE3DModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = HardSigmoid() if use_hs else nn.Sigmoid()

    def forward(self, x):
        """forward"""
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class SE2DModule(nn.Module):
    """SE 2D"""
    def __init__(self, channel, reduction=16, use_hs=False):
        super(SE2DModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            HardSigmoid() if use_hs else nn.Sigmoid()
        )

    def forward(self, x):
        """forward"""
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
