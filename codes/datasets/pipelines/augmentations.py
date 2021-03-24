"""data augmentations"""
import math
import random

import cv2
import mmcv
import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module
class Resize(object):
    """Resize images to a specific size.

    Required keys are "img_group", added or modified keys are "img_group",
    "img_shape", "keep_ratio", "scale_factor" and "resize_size".

    Attributes:
        scale (int | Tuple[int]): Target spatial size (h, w).
        keep_ratio (bool): If set to True, Images will be resized without
            changing the aspect ratio. Otherwise, it will resize images to a
            given size.
        interpolation (str): Algorithm used for interpolation:
            "nearest" | "bilinear". Default: "bilinear".
    """

    def __init__(self, scale, keep_ratio=True, interpolation='bilinear'):
        if isinstance(scale, (float, int)):
            if scale <= 0:
                raise ValueError(
                    'Invalid scale {}, must be positive.'.format(scale))
        self.scale = scale
        self.keep_ratio = keep_ratio
        self.interpolation = interpolation

    def __call__(self, results):
        img_group = results['img_group']
        if self.keep_ratio:
            tuple_list = [
                mmcv.imrescale(img, self.scale, return_scale=True)
                for img in img_group
            ]
            img_group, scale_factors = list(zip(*tuple_list))
            self.scale_factor = scale_factors[0]
        else:
            tuple_list = [
                mmcv.imresize(img, self.scale, return_scale=True)
                for img in img_group
            ]
            img_group, w_scales, h_scales = list(zip(*tuple_list))
            self.scale_factor = np.array(
                [w_scales[0], h_scales[0], w_scales[0], h_scales[0]],
                dtype=np.float32)

        results['img_group'] = img_group
        results['img_shape'] = img_group[0].shape
        results['keep_ratio'] = self.keep_ratio
        results['scale_fatcor'] = self.scale_factor

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(scale={}, keep_ratio={}, interpolation={})'. \
            format(self.scale, self.keep_ratio, self.interpolation)
        return repr_str


@PIPELINES.register_module
class MultiScaleCrop(object):
    """Crop an image with a randomly selected scale.

    Randomly select the w and h scales from a list of scales. Scale of 1 means
    the base size, which is the minimal of image weight and height. The scale
    level of w and h is controlled to be smaller than a certain value to
    prevent too large or small aspect ratio.
    Required keys are "img_group", added or modified keys are "img_group",
    "crop_bbox", "img_shape" and "scales".

    Attributes:
        input_size (int | tuple[int]): (w, h) of network input.
        scales (list[float]): Weight and height scales to be selected.
        max_distort (int): Maximum gap of w and h scale levels.
            Default: 1.
        fix_crop (bool): If set to False, the cropping bbox will be randomly
            sampled, otherwise it will be sampler from 5 fixed regions:
            "upper left", "upper right", "lower left", "lower right", "center"
            Default: True.
    """

    def __init__(self,
                 input_size,
                 scales=None,
                 max_distort=1,
                 fix_crop=True,
                 more_fix_crop=True):
        self.input_size = input_size if not isinstance(input_size, int) \
            else (input_size, input_size)
        self.scales = scales if scales is not None else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.interpolation = 'bilinear'

    def __call__(self, results):
        img_group = results['img_group']
        img_h, img_w = img_group[0].shape[:2]

        (crop_w, crop_h), (offset_w, offset_h) = self._sample_crop_size(
            (img_w, img_h))
        box = np.array([offset_w, offset_h, offset_w +
                        crop_w - 1, offset_h + crop_h - 1])
        crop_img_group = [mmcv.imcrop(img, box) for img in img_group]
        ret_img_group = [mmcv.imresize(
            img, (self.input_size[0], self.input_size[1]),
            interpolation=self.interpolation)
            for img in crop_img_group]

        results['crop_bbox'] = box

        results['img_group'] = ret_img_group
        results['img_shape'] = ret_img_group[0].shape
        results['scales'] = self.scales
        return results

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(
            x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(
            x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(
                image_w, image_h, crop_pair[0], crop_pair[1])

        return (crop_pair[0], crop_pair[1]), (w_offset, h_offset)

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(
            self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        """fill fix offset"""
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(input_size={}, scales={}, max_distort={}, fix_crop={}'
                     'more_fix_crop={})').format(self.input_size, self.scales,
                                                 self.max_distort,
                                                 self.fix_crop,
                                                 self.more_fix_crop)
        return repr_str


@PIPELINES.register_module
class Flip(object):
    """Flip the input images with a probability.

    Reverse the order of elements in the given imgs with a specific direction.
    The shape of the imgs is preserved, but the elements are reordered.
    Required keys are "img_group", added or modified keys are "img_group"
    and "flip_direction".

    Attributes:
         direction (str): Flip imgs horizontally or vertically. Options are
            "horiziontal" | "vertival". Default: "horizontal".
    """

    def __init__(self, flip_ratio=0.5, direction='horizontal'):
        assert direction in ['horizontal', 'vertical']
        self.flip_ratio = flip_ratio
        self.direction = direction

    def __call__(self, results):
        img_group = results['img_group']
        flip = True if np.random.rand() < self.flip_ratio else False
        if flip:
            img_group = [mmcv.imflip(img, self.direction)
                         for img in img_group]
        if results['modality'] == 'Flow':
            for i in range(0, len(img_group), 2):
                img_group[i] = mmcv.iminvert(img_group[i])

        results['flip'] = flip
        results['flip_direction'] = self.direction
        results['img_group'] = img_group

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(flip_ratio={}, direction={})'.format(
            self.flip_ratio, self.direction)
        return repr_str


@PIPELINES.register_module
class ColorJitter(object):
    """color jitter"""
    def __init__(self, color_space_aug=False,
                 alphastd=0.1, eigval=None, eigvec=None):
        if eigval is None:
            # note that the data range should be [0, 255]
            self.eigval = np.array([55.46, 4.794, 1.148])
        if eigvec is None:
            self.eigvec = np.array([[-0.5675, 0.7192, 0.4009],
                                    [-0.5808, -0.0045, -0.8140],
                                    [-0.5836, -0.6948, 0.4203]])
        self.alphastd = alphastd
        self.color_space_aug = color_space_aug

    @staticmethod
    def brightnetss(img, delta):
        """brightnetss"""
        if random.uniform(0, 1) > 0.5:
            # delta = np.random.uniform(-32, 32)
            delta = np.array(delta).astype(np.float32)
            img = img + delta
            # img_group = [img + delta for img in img_group]
        return img

    @staticmethod
    def contrast(img, alpha):
        """contrast"""
        if random.uniform(0, 1) > 0.5:
            # alpha = np.random.uniform(0.6,1.4)
            alpha = np.array(alpha).astype(np.float32)
            img = img * alpha
            # img_group = [img * alpha for img in img_group]
        return img

    @staticmethod
    def saturation(img, alpha):
        """saturation"""
        if random.uniform(0, 1) > 0.5:
            # alpha = np.random.uniform(0.6,1.4)
            gray = img * np.array([0.299, 0.587, 0.114]).astype(np.float32)
            gray = np.sum(gray, 2, keepdims=True)
            gray *= (1.0 - alpha)
            img = img * alpha
            img = img + gray
        return img

    @staticmethod
    def hue(img, alpha):
        """hue"""
        if random.uniform(0, 1) > 0.5:
            # alpha = random.uniform(-18, 18)
            u = np.cos(alpha * np.pi)
            w = np.sin(alpha * np.pi)
            bt = np.array([[1.0, 0.0, 0.0],
                           [0.0, u, -w],
                           [0.0, w, u]])
            tyiq = np.array([[0.299, 0.587, 0.114],
                             [0.596, -0.274, -0.321],
                             [0.211, -0.523, 0.311]])
            ityiq = np.array([[1.0, 0.956, 0.621],
                              [1.0, -0.272, -0.647],
                              [1.0, -1.107, 1.705]])
            t = np.dot(np.dot(ityiq, bt), tyiq).T
            t = np.array(t).astype(np.float32)
            img = np.dot(img, t)
            # img_group = [np.dot(img, t) for img in img_group]
        return img

    def __call__(self, results):
        img_group = results['img_group']
        if self.color_space_aug:
            bright_delta = np.random.uniform(-32, 32)
            contrast_alpha = np.random.uniform(0.6, 1.4)
            saturation_alpha = np.random.uniform(0.6, 1.4)
            hue_alpha = random.uniform(-18, 18)
            out = []
            for img in img_group:
                img = self.brightnetss(img, delta=bright_delta)
                if random.uniform(0, 1) > 0.5:
                    img = self.contrast(img, alpha=contrast_alpha)
                    img = self.saturation(img, alpha=saturation_alpha)
                    img = self.hue(img, alpha=hue_alpha)
                else:
                    img = self.saturation(img, alpha=saturation_alpha)
                    img = self.hue(img, alpha=hue_alpha)
                    img = self.contrast(img, alpha=contrast_alpha)
                out.append(img)
            img_group = out

        alpha = np.random.normal(0, self.alphastd, size=(3,))
        rgb = np.array(np.dot(self.eigvec * alpha, self.eigval)
                       ).astype(np.float32)
        bgr = np.expand_dims(np.expand_dims(rgb[::-1], 0), 0)
        img_group = [img + bgr for img in img_group]
        results['img_group'] = img_group
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(color_space_aug={}, alphastd={})'.format(
            self.color_space_aug, self.alphastd)
        return repr_str


@PIPELINES.register_module
class Normalize(object):
    """Normalize images with the given mean and std value.

    Required keys are "img_group", added or modified keys are
    "img_group" and "img_norm_cfg".

    Attributes:
        mean (np.ndarray): Mean values of different channels.
        std (np.ndarray): Std values of different channels.
        to_rgb (bool): Whether to convert channels from BGR to RGB.
    """

    def __init__(self, mean, std, div_255=False, to_rgb=False):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.div_255 = div_255
        self.to_rgb = to_rgb

    # def imnormalize(self, img, mean, std, to_rgb=True):
    #     img = img.astype(np.float32)
    #     if to_rgb:
    #         cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
    #     return (img - mean) / std

    def imnormalize(self, img, mean, std, to_rgb=True):
        """imnormalize"""
        img = np.float32(img) if img.dtype != np.float32 else img.copy()
        mean = np.float64(mean.reshape(1, -1))
        stdinv = 1 / np.float64(std.reshape(1, -1))
        if to_rgb:
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
        cv2.subtract(img, mean, img)  # inplace, faster
        cv2.multiply(img, stdinv, img)  # inplace, faster
        return img

    def __call__(self, results):
        img_group = results['img_group']
        if self.div_255:
            img_group = [np.float32(img) / 255 for img in img_group]

        img_group = [self.imnormalize(
            img, self.mean, self.std, self.to_rgb) for img in img_group]

        results['img_group'] = img_group
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std,
            div_255=self.div_255, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(mean={}, std={}, to_rgb={})'.format(
            self.mean, self.std, self.to_rgb)
        return repr_str


@PIPELINES.register_module
class Pad(object):
    """Pad images to ensure each edge to be multiple to some number.
    Required keys are "img_group", added or modified keys are
    "img_group" and "divisor".
    Attributes:
        divisor (int): Padded image edges will be multiple to divisor.

    Returns:
        ndarray: The padded image.
    """

    def __init__(self, divisor):
        self.divisor = divisor

    def __call__(self, results):
        img_group = [mmcv.impad_to_multiple(
            img, self.divisor) for img in results['img_group']]

        results['img_group'] = img_group
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(divisor={})'.format(self.divisor)
        return repr_str


@PIPELINES.register_module
class CenterCrop(object):
    """Crop the center area from images.

    Required keys are "img_group", added or modified keys are "img_group",
    "crop_bbox" and "img_shape".

    Attributes:
        crop_size(tuple[int]): (w, h) of crop size.
    """

    def __init__(self, crop_size=224):
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        else:
            self.crop_size = crop_size
        assert mmcv.is_tuple_of(self.crop_size, int)

    def __call__(self, results):
        img_group = results['img_group']

        img_h, img_w = img_group[0].shape[:2]
        crop_w, crop_h = self.crop_size
        x1 = (img_w - crop_w) // 2
        y1 = (img_h - crop_h) // 2
        box = np.array([x1, y1, x1 + crop_w - 1, y1 + crop_h - 1])
        results['img_group'] = [mmcv.imcrop(img, box) for img in img_group]
        results['crop_bbox'] = box
        results['img_shape'] = results['img_group'][0].shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(crop_size={})'.format(self.crop_size)
        return repr_str


@PIPELINES.register_module
class ThreeCrop(object):
    """Crop images into three crops.

    Crop the images equally into three crops with equal intervals along the
    shorter side.
    Required keys are "img_group", added or modified keys are "img_group",
    "crop_bbox" and "img_shape".

    Attributes:
        crop_size(tuple[int]): (w, h) of crop size.
    """

    def __init__(self, crop_size):
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        else:
            self.crop_size = crop_size
        assert mmcv.is_tuple_of(self.crop_size, int)

    def __call__(self, results):
        img_group = results['img_group']
        img_h, img_w = img_group[0].shape[:2]
        crop_w, crop_h = self.crop_size
        # assert crop_h == img_h or crop_w == img_w

        if crop_h == img_h:
            w_step = (img_w - crop_w) // 2
            offsets = [
                (0, 0),  # left
                (2 * w_step, 0),  # right
                (w_step, 0),  # middle
            ]
        elif crop_w == img_w:
            h_step = (img_h - crop_h) // 2
            offsets = [
                (0, 0),  # top
                (0, 2 * h_step),  # down
                (0, h_step),  # middle
            ]
        else:
            w_step = (img_w - crop_w) // 4
            h_step = (img_h - crop_h) // 4

            offsets = list()
            offsets.append((0 * w_step, 2 * h_step))  # left
            offsets.append((4 * w_step, 2 * h_step))  # right
            offsets.append((2 * w_step, 2 * h_step))  # center            

        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = list()
            flip_group = list()
            for i, img in enumerate(img_group):
                crop = mmcv.imcrop(img, np.array(
                    [o_w, o_h, o_w + crop_w - 1, o_h + crop_h - 1]))
                normal_group.append(crop)
                flip_crop = mmcv.imflip(crop)

                if results['modality'] == 'Flow' and i % 2 == 0:
                    flip_group.append(mmcv.iminvert(flip_crop))
                else:
                    flip_group.append(flip_crop)

            oversample_group.extend(normal_group)

        results['img_group'] = oversample_group
        results['crop_bbox'] = None
        results['img_shape'] = results['img_group'][0].shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(crop_size={})'.format(self.crop_size)
        return repr_str


@PIPELINES.register_module
class TenCrop(object):
    """Crop the images into 10 crops (corner + center + flip).

    Crop the four corners and the center part of the image with the same
    given crop_size, and flip it horizontally.
    Required keys are "img_group", added or modified keys are "img_group",
    "crop_bbox" and "img_shape".

    Attributes:
        crop_size(tuple[int]): (w, h) of crop size.
    """

    def __init__(self, crop_size=224):
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        else:
            self.crop_size = crop_size
        assert mmcv.is_tuple_of(self.crop_size, int)

    def __call__(self, results):
        img_group = results['img_group']
        img_h, img_w = img_group[0].shape[:2]
        crop_w, crop_h = self.crop_size

        offsets = MultiScaleCrop.fill_fix_offset(
            False, img_w, img_h, crop_w, crop_h)
        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = list()
            flip_group = list()
            for i, img in enumerate(img_group):
                crop = mmcv.imcrop(img, np.array(
                    [o_w, o_h, o_w + crop_w - 1, o_h + crop_h - 1]))
                normal_group.append(crop)
                flip_crop = mmcv.imflip(crop)

                if results['modality'] == 'Flow' and i % 2 == 0:
                    flip_group.append(mmcv.iminvert(flip_crop))
                else:
                    flip_group.append(flip_crop)

            oversample_group.extend(normal_group)
            oversample_group.extend(flip_group)
        results['img_group'] = oversample_group
        results['crop_bbox'] = None
        results['img_shape'] = results['img_group'][0].shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(crop_size={})'.format(self.crop_size)
        return repr_str


@PIPELINES.register_module
class RandomResizedCrop(object):
    """random resize crop"""
    def __init__(self, input_size, scale=(0.08, 1.0),
                 ratio=(3. / 4., 4. / 3.)):
        if isinstance(input_size, int):
            self.input_size = (input_size, input_size)
        else:
            self.input_size = input_size
        assert mmcv.is_tuple_of(self.input_size, int)
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio
                           cropped
        Returns:
            tuple: params (i, j), (h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        for attempt in range(10):
            area = img.shape[0] * img.shape[1]
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.shape[0] and h <= img.shape[1]:
                i = random.randint(0, img.shape[1] - h)
                j = random.randint(0, img.shape[0] - w)
                return (i, j), (h, w)

        # Fallback
        w = min(img.shape[0], img.shape[1])
        i = (img.shape[1] - w) // 2
        j = (img.shape[0] - w) // 2
        return (i, j), (w, w)

    def __call__(self, results):
        """
        Args:
            clip (list of PIL Image): list of Image to be cropped and resized.
        Returns:
            list of PIL Image: Randomly cropped and resized image.
        """
        img_group = results['img_group']

        (x1, y1), (crop_h, crop_w) = self.get_params(
            img_group[0], self.scale, self.ratio)
        box = np.array([x1, y1, x1 + crop_w - 1, y1 + crop_h - 1], dtype=np.float32)

        results['img_group'] = [mmcv.imresize(mmcv.imcrop(
            img, box), self.input_size) for img in img_group]
        results['crop_bbox'] = box
        results['img_shape'] = results['img_group'][0].shape
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(input_size={})'.format(self.input_size)
        return repr_str


@PIPELINES.register_module
class RandomRescaledCrop(object):
    """random rescale crop"""
    def __init__(self, input_size, scale=(256, 320)):
        if isinstance(input_size, int):
            self.input_size = (input_size, input_size)
        else:
            self.input_size = input_size
        assert mmcv.is_tuple_of(self.input_size, int)
        self.scale = scale

    def __call__(self, results):
        img_group = results['img_group']
        shortedge = float(random.randint(*self.scale))

        w, h, _ = img_group[0].shape
        scale = max(shortedge / w, shortedge / h)
        img_group = [mmcv.imrescale(img, scale) for img in img_group]
        w, h, _ = img_group[0].shape
        w_offset = random.randint(0, w - self.input_size[0])
        h_offset = random.randint(0, h - self.input_size[1])

        box = np.array([w_offset, h_offset,
                        w_offset + self.input_size[0] - 1,
                        h_offset + self.input_size[1] - 1],
                       dtype=np.float32)
        results['img_group'] = [
            img[w_offset: w_offset + self.input_size[0],
                h_offset: h_offset + self.input_size[1]] for img in img_group]
        results['crop_bbox'] = box
        results['img_shape'] = results['img_group'][0].shape
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(input_size={})'.format(self.input_size)
        return repr_str
