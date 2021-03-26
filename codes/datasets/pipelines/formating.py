"""formating"""
from collections.abc import Sequence

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC

from ..builder import PIPELINES


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError('type {} cannot be converted to tensor.'.format(
            type(data)))


@PIPELINES.register_module
class ToTensor(object):
    """To Tensor"""
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            results[key] = to_tensor(results[key])
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(keys={})'.format(self.keys)


@PIPELINES.register_module
class ImageToTensor(object):
    """Image To tensor"""
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            results[key] = to_tensor(results[key].transpose(2, 0, 1))
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(keys={})'.format(self.keys)


@PIPELINES.register_module
class Transpose(object):
    """Transpose"""
    def __init__(self, keys, order):
        self.keys = keys
        self.order = order

    def __call__(self, results):
        for key in self.keys:
            results[key] = results[key].transpose(self.order)
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(keys={}, order={})'.format(
            self.keys, self.order)


@PIPELINES.register_module
class Collect(object):
    """Collect data from the loader relevant to the specific task.

    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img_group", "gt_labels".

    The "img_meta" item is always populated.  The contents of the "img_meta"
    dictionary depends on "meta_keys". By default this includes:

        - "img_shape": shape of the image input to the network as a tuple
                (h, w, c).  Note that images may be zero padded on the
                bottom/right, if the batch tensor is larger than this shape.

        - "scale_factor": a float indicating the preprocessing scale

        - "flip": a boolean indicating if image flip transform was used

        - "filename": path to the image file

        - "ori_shape": original shape of the image as a tuple (h, w, c)

        - "img_norm_cfg": a dict of normalization information:
            - mean - per channel mean subtraction
            - std - per channel std divisor
            - div_255 - bool indicating if pixel value div 255
            - to_rgb - bool indicating if bgr was converted to rgb
    """

    def __init__(self,
                 keys,
                 meta_keys=('label', 'ori_shape', 'img_shape',
                            'modality', 'img_norm_cfg')):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        data = {}
        if len(self.meta_keys) != 0:
            img_meta = {}
            for key in self.meta_keys:
                img_meta[key] = results[key]
            data['img_meta'] = DC(img_meta, cpu_only=True)

        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return self.__class__.__name__ + '(keys={}, meta_keys={})'.format(
            self.keys, self.meta_keys)


@PIPELINES.register_module
class FormatShape(object):
    """Format final imgs shape to the given input_format

    Required keys are "img_group", "num_clips" and "clip_len",
    added or modified keys are "img_group" and "input_shape".

    Attributes:
        input_format (str): define the final imgs format.

    """

    def __init__(self, input_format='NCHW'):
        assert input_format in ['NCHW', 'NCTHW']
        # final input_format is BNCHW OR BNCTHW
        self.input_format = input_format

    def __call__(self, results):
        img_group = results['img_group']
        # transpose
        if results['modality'] == 'Flow':
            assert len(img_group[0].shape) == 2
            img_group = [np.stack((flow_x, flow_y), axis=2)
                         for flow_x, flow_y in zip(
                             img_group[0::2], img_group[1::2])]
        img_group = [img.transpose(2, 0, 1) for img in img_group]
        # Stack into numpy.array
        img_group = np.stack(img_group, axis=0)
        # [M x C x H x W]

        # M = 1 * N_oversample * N_clips * L
        num_clips = results['num_clips']
        clip_len = results['clip_len']
        if self.input_format == 'NCTHW':
            if clip_len == 1 and num_clips > 1:
                # uniform sampling, num_clips mean clip_len
                img_group = img_group.reshape(
                    (-1, num_clips) + img_group.shape[1:])
                # N_over x N_clips x C x H x W
                img_group = np.transpose(img_group, (0, 2, 1, 3, 4))
                # N_over x C x N_clips x H x W
            else:
                img_group = img_group.reshape(
                    (-1, num_clips, clip_len) + img_group.shape[1:])
                # N_over x N_clips x L x C x H x W
                img_group = np.transpose(img_group, (0, 1, 3, 2, 4, 5))
                # N_over x N_clips x C x L x H x W
                img_group = img_group.reshape((-1, ) + img_group.shape[2:])
                # M' x C x L x H x W
                # M' = N_over x N_clips
        results['img_group'] = img_group
        results['input_shape'] = img_group.shape
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(input_format={})'.format(self.input_format)
