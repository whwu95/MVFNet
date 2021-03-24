"""backbone init"""
from .bninception import BNInception
from .inception_v1_i3d import InceptionV1_I3D
from .mobilenet_v2 import InvertedResidual, MobileNetV2
from .resnet import ResNet
from .resnet_i3d import ResNet_I3D
from .resnet_i3d_slowfast import ResNet_I3D_SlowFast
from .resnet_r3d import ResNet_R3D 
from .resnet_x3d import ResNet_X3D 

# from .resnet_s3d import ResNet_S3D

__all__ = [
    'InvertedResidual', 'MobileNetV2',
    'BNInception',
    'ResNet',
    'InceptionV1_I3D',
    'ResNet_I3D',
    # 'ResNet_S3D',
    'ResNet_I3D_SlowFast',
    'ResNet_R3D',
    'ResNet_X3D',
]
