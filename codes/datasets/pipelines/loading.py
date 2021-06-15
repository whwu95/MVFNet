"""loading"""
import os.path as osp
import mmcv
import numpy as np
from ...utils import FileClient, get_root_logger
from ..builder import PIPELINES
logger = get_root_logger()
# from io import StringIO, BytesIO
# import collections
# from PIL import Image
@PIPELINES.register_module
class SampleFrames(object):
    """Sample frames from the video.
    Pass data by dict "results". Required keys are "filename",
    added or modified keys are "total_frames",
    "frame_inds", "frame_interval" and "num_clips".
    Attributes:
        clip_len (int): Frames of each sampled output clip.
        frame_interval (int): Temporal interval of adjacent sampled frames.
        num_clips (int): Number of clips to be sampled.
        temporal_jitter (bool): Whether to apply temporal jittering.
    """

    def __init__(self,
                 clip_len,
                 frame_interval=1,
                 num_clips=1,
                 temporal_jitter=False,
                 sth_samples=1):
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.temporal_jitter = temporal_jitter
        self.sth_samples = sth_samples  # for test sth-sth
        # self.decode_type = decode_type # ['rawframes', 'opencv', 'decord', 'pyav']

    def _sample_clips(self, num_frames):
        """Choose frame indices for the video in training phase.
        Calculate the average interval for selected frames, and randomly
        shift them within offsets between [0, avg_interval]. If the total
        number of frames is smaller than clips num or origin frames length,
        it will return all zero indices.
        Args:
            num_frames (int): Total number of frame in the video.
        Returns:
            np.ndarray: Sampled frame indices (load image need to add 1).
        """
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips
        if avg_interval > 0:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = base_offsets + np.random.randint(
                avg_interval, size=self.num_clips)
        elif num_frames > max(self.num_clips, ori_clip_len):
            clip_offsets = np.sort(
                np.random.randint(
                    num_frames - ori_clip_len + 1, size=self.num_clips))
        else:
            clip_offsets = np.zeros((self.num_clips, ))
        return clip_offsets

    def _test_sample_clips(self, num_frames):
        ori_clip_len = self.clip_len * self.frame_interval
        tick = (num_frames - ori_clip_len + 1) / float(self.num_clips)
        if self.sth_samples == 1:
            if tick > 0:
                clip_offsets = np.array([int(tick / 2.0 + tick * x)
                                         for x in range(self.num_clips)])
            else:
                clip_offsets = np.zeros((self.num_clips, ))
        elif self.sth_samples == 2:
            clip_offsets = np.array(
                [int(tick / 2.0 + tick * x) for x in range(self.num_clips)] + [
                    int(tick * x) for x in range(self.num_clips)])
        elif self.sth_samples == 10:
            offsets = []
            for i in range(10):
                offsets += self._sample_clips(num_frames).tolist()
            clip_offsets = np.array(offsets)
        else:
            clip_offsets = []
            offsets = np.array(
                [int(tick / 2.0 + tick * x) for x in range(self.num_clips)])
            clip_offsets.append(offsets)
            avg_duration = (
                num_frames - ori_clip_len + 1) // float(self.num_clips)
            for i in range(self.sth_samples - 1):
                offsets = np.multiply(
                    list(range(self.num_clips)), avg_duration) + np.random.randint(avg_duration, size=self.num_clips)
                clip_offsets.append(offsets)
            clip_offsets = np.stack(clip_offsets).reshape(-1)
        return clip_offsets

    def _get_frame_inds(self, total_frames, results):
        if results['test_mode']:
            clip_offsets = self._test_sample_clips(total_frames)
        else:
            clip_offsets = self._sample_clips(total_frames)
        # size: [num_clip clip_len]
        frame_inds = clip_offsets[:, None] + np.arange(
            self.clip_len)[None, :] * self.frame_interval
        if self.temporal_jitter:
            perframe_offsets = np.random.randint(
                self.frame_interval, size=self.clip_len)
            # [num_clip clip_len] + [1 clip_len]
            #  each clip add a same jitter offset
            frame_inds += perframe_offsets[None, :]
        # size: clip_len * num_clip
        frame_inds = np.concatenate(frame_inds)
        # if temporal_jitter, mabye out of range
        # frame_inds = np.mod(frame_inds, total_frames)
        frame_inds = np.minimum(frame_inds, total_frames - 1).astype(np.int)
        return frame_inds

    def __call__(self, results):
        if 'total_frames' not in results:
            # TODO: find a better way to get the total frames number for video
            video_reader = mmcv.VideoReader(results['filename'])
            # import decord
            # video_reader = decord.VideoReader(results['filename'])
            total_frames = len(video_reader)
            results['total_frames'] = total_frames
        else:
            total_frames = results['total_frames']

        results['frame_inds'] = self._get_frame_inds(total_frames, results)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = self.num_clips
        results['sth_samples'] = self.sth_samples
        return results


@PIPELINES.register_module
class PyAVDecode(object):
    """Using pyav to decode the video.
    PyAV: https://github.com/mikeboers/PyAV
    Required keys are "filename" and "frame_inds",
    added or modified keys are "img_group" and "ori_shape".
    Attributes:
        multi_thread (bool): If set to True, it will
            apply multi thread processing.
    """

    def __init__(self, multi_thread=True, accurate=False):
        self.multi_thread = multi_thread
        self.accurate = accurate

    def frame_generator(self, container, stream):
        """frame generator
        Args:
            container ([type]): [description]
            stream ([type]): [description]
        Returns:
            [type]: [description]
        """
        for packet in container.demux(stream):
            for frame in packet.decode():
                if frame:
                    return frame.to_ndarray(format='rgb24')

    def __call__(self, results):
        try:
            import av
        except ImportError:
            raise ImportError('Please run "conda install av -c conda-forge" '
                              'or "pip install av" to install PyAV first.')
        av.logging.set_level(5)
        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        try:
            container = av.open(results['filename'])
            stream = container.streams.video[0]
            if self.multi_thread:
                stream.thread_type = 'AUTO'
            # check duration
            try:
                duration = stream.duration * stream.time_base
            except TypeError:
                duration = container.duration / av.time_base
            if duration <= 0:
                raise IOError("Video stream 0 in {} has zero length.".format(
                    results['filename']))

            frame_count = stream.frames
            max_inds = max(results['frame_inds'])
            if max_inds > frame_count:
                frame_inds = [idx %
                              frame_count for idx in results['frame_inds']]
            else:
                frame_inds = results['frame_inds']
            img_group = list()
            if self.accurate:  # for accurate seeking
                i = 0
                for frame in container.decode(video=0):
                    # set max indice to make early stop
                    if i > max_inds + 1:
                        break
                    # some other formats gray16be, bgr24, rgb24
                    img_group.append(frame.to_ndarray(format='rgb24'))
                    i += 1

                # the available frame in pyav may be less than its length, which may raise error
                results['img_group'] = [img_group[i %
                                                  len(img_group)] for i in results['frame_inds']]
            else:   # for fast seeking (not accurate)
                for idx in frame_inds.tolist():
                    pts_scale = stream.average_rate * stream.time_base
                    frame_pts = int(idx / pts_scale)
                    container.seek(frame_pts, any_frame=False,
                                   backward=True, stream=stream)
                    frame = self.frame_generator(container, stream)
                    if frame is not None:
                        img_group.append(frame)
                    else:
                        img_group.append(img_group[-1])
                results['img_group'] = img_group
            container.close()

            results['ori_shape'] = results['img_group'][0].shape[:2]
        except Exception as e:
            logger.info("Failed to decode {} with exception: {}".format(
                results['filename'], e))
            return None

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(multi_thread={})'.format(self.multi_thread)


@PIPELINES.register_module
class PIMSDecode(object):
    """Using PIMS to decode the video.
    PIMS: https://github.com/soft-matter/pims
    Required keys are "filename" and "frame_inds",
    added or modified keys are "img_group" and "ori_shape".
    Attributes:
        multi_thread (bool): If set to True, it will
            apply multi thread processing.
    """

    def __init__(self, indexed=True):
        self.indexed = indexed

    def __call__(self, results):
        try:
            import pims
        except ImportError:
            raise ImportError('Please run "conda install pims -c conda-forge" '
                              'or "pip install pims" to install pims first.')
        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        try:
            if self.indexed:  # faster than pyav seek (accurate)
                video = pims.PyAVReaderIndexed(results['filename'])
            else:
                # faster, but something wrong with pytorch dataloader
                video = pims.PyAVReaderTimed(results['filename'])

            frame_count = len(video)
            max_inds = max(results['frame_inds'])
            if max_inds > frame_count:
                frame_inds = [idx %
                              frame_count for idx in results['frame_inds']]
            else:
                frame_inds = results['frame_inds']
            img_group = video[frame_inds]
            results['img_group'] = img_group
            results['ori_shape'] = results['img_group'][0].shape[:2]
        except Exception as e:
            logger.info("Failed to decode {} with exception: {}".format(
                results['filename'], e))
            return None

        return results


@PIPELINES.register_module
class DecordDecode(object):
    """Using decord to decode the video.
    Decord: https://github.com/zhreshold/decord
    Required keys are "filename" and "frame_inds",
    added or modified keys are "img_group" and "ori_shape".
    Attributes:
        num_threads (int): multi thread processing.
        accurate (bool): random access patterns
    """

    def __init__(self, num_threads=0, accurate=True):
        self.num_threads = num_threads
        self.accurate = accurate

    def __call__(self, results):
        try:
            import decord
        except ImportError:
            raise ImportError(
                'Please run "pip install decord" to install Decord first.')
        decord.logging.set_level(5)
        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])
        try:
            container = decord.VideoReader(
                results['filename'], num_threads=self.num_threads)
            num_frames = len(container)  # decord num_frames
            frame_inds = [idx % num_frames for idx in results['frame_inds']]
            # Generate frame index mapping in order
            # frame_dict = {idx: container[idx % num_frames].asnumpy() for idx in np.unique(frame_inds)}
            # img_group = [frame_dict[idx] for idx in frame_inds]

            if self.accurate:
                img_group = container.get_batch(frame_inds).asnumpy()
            else:
                # faster, however always return I-FRAME
                container.seek(0)
                img_group = []
                for idx in frame_inds:
                    container.seek(idx)
                    frame = container.next()
                    img_group.append(frame.asnumpy())

            del container
            results['img_group'] = img_group
            results['ori_shape'] = img_group[0].shape
            results['img_shape'] = img_group[0].shape
        except Exception as e:
            logger.info("Failed to decode {} with exception: {}".format(
                results['filename'], e))
            return None
        return results


@PIPELINES.register_module
class OpenCVDecode(object):
    """Using OpenCV to decode the video.
    Required keys are "filename" and "frame_inds",
    added or modified keys are "img_group" and "ori_shape".
    """

    def __call__(self, results):
        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])
        try:
            container = mmcv.VideoReader(results['filename'])
            img_group = list()
            for frame_ind in results['frame_inds']:
                cur_frame = container[frame_ind]
                try:
                    cur_frame = container[frame_ind]
                except IndexError:
                    logger.info(results['filename'],
                                frame_ind, results['total_frames'])
                # last frame may be None in OpenCV
                while isinstance(cur_frame, type(None)):
                    frame_ind -= 1
                    cur_frame = container[frame_ind]
                img_group.append(cur_frame)
            # img_group = np.array(img_group)
            # The default channel order of OpenCV is BGR, thus we change it to RGB
            # img_group = img_group[:, :, :, ::-1]
            # imgs = imgs.transpose([0, 3, 1, 2])
            results['img_group'] = img_group
            results['ori_shape'] = img_group[0].shape
        except Exception as e:
            logger.info("Failed to decode {} with exception: {}".format(
                results['filename'], e))
            return None
        return results


@PIPELINES.register_module
class PklLoader(object):
    """Using pickle to loader pkl file.
    Required keys are "filename" and "frame_inds",
    added or modified keys are "img_group" and "ori_shape".
    """

    def _pil_loader(self, buf, usegray=False):
        # print(type(buf))
        if isinstance(buf, bytes):
            img = mmcv.imfrombytes(buf, 'color')
            # img = Image.open(BytesIO(buf))
            # tempbuff = BytesIO()
            # tempbuff.write(buf)
            # tempbuff.seek(0)
            # img = Image.open(tempbuff)
        # elif isinstance(buf,collections.Sequence):
        #      img = Image.open(BytesIO(buf[-1]))
        else:
            logger.info('Maybe something wrong')
        # return img.convert('L') if usegray else img.convert('RGB')
        return np.array(img)

    def __call__(self, results):
        try:
            import _pickle as pickle
        except ImportError:
            raise ImportError(
                'Please run "pip install _pickle" to install _pickle first.')
        container = pickle.load(open(results['filename'], 'rb'))
        img_group = list()
        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])
        for frame_idx in results['frame_inds']:
            cur_frame = self._pil_loader(container[frame_idx])
            img_group.append(cur_frame)
            # img_group.append(cur_frame[:, :, ::-1])
        results['img_group'] = img_group
        results['ori_shape'] = img_group[0].shape
        return results


@PIPELINES.register_module
class FrameSelector(object):
    """Select raw frames with given indices
    Required keys are "file_dir", "filename_tmpl" and "frame_inds",
    added or modified keys are "img_group" and "ori_shape".
    Attributes:
        io_backend (str): io backend where frames are store.
    """

    def __init__(self, io_backend='disk', **kwargs):
        self.io_backend = io_backend
        self.file_client = FileClient(self.io_backend, **kwargs)
        self.backup = None

    def _load_image(self, filepath, flag='color'):
        value_buf = self.file_client.get(filepath)
        try:
            cur_frame = mmcv.imfrombytes(value_buf, flag)
        except Exception:
            logger.info('imfrombytes error, reload backup')
            cur_frame = self.backup
        # cur_frame = mmcv.imread(filepath)
        return cur_frame

    def __call__(self, results):
        directory = results['filename']
        filename_tmpl = results['filename_tmpl']
        imgs = list()
        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])
        for frame_idx in results['frame_inds']:
            if results['modality'] in ['RGB', 'RGBDiff']:
                filepath = osp.join(
                    directory, filename_tmpl.format(frame_idx + 1))
                cur_frame = [self._load_image(filepath)]
            elif results['modality'] == 'Flow':
                x_imgs = self._load_image(
                    osp.join(
                        directory, filename_tmpl.format(
                            'x', frame_idx + 1)), flag='grayscale')
                y_imgs = self._load_image(
                    osp.join(
                        directory, filename_tmpl.format(
                            'y', frame_idx + 1)), flag='grayscale')
                cur_frame = [x_imgs, y_imgs]
            else:
                raise ValueError(
                    'Not implemented yet; modality should be '
                    '["RGB", "RGBDiff", "Flow"]')
            imgs.extend(cur_frame)
            if self.backup is None:
                self.backup = cur_frame
        # # [num c h w]
        # imgs = np.array(imgs)
        # imgs = imgs.transpose([0, 3, 1, 2])
        results['img_group'] = imgs
        # [h w c]
        results['ori_shape'] = imgs[0].shape
        return results
