"""raw frames dataset"""
import copy
import os.path as osp

from .base import BaseDataset
from .builder import DATASETS


@DATASETS.register_module
class RawFramesDataset(BaseDataset):
    """RawFrames dataset for action recognition.

    The dataset loads raw frames and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    the directory to frames of a video, total frames of the video and
    the label of a video, which are split with a whitespace.
    Example of a annotation file:

    ```
    some/directory-1 163 1
    some/directory-2 122 1
    some/directory-3 258 2
    some/directory-4 234 2
    some/directory-5 295 3
    some/directory-6 121 3
    ```

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        data_root (str): Path to a directory where videos are held.
        filename_tmpl (str): Template for each filename.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root=None,
                 test_mode=False,
                 filename_tmpl='img_{:05}.jpg',
                 modality='RGB'):
        super(RawFramesDataset, self).__init__(ann_file, pipeline,
                                               data_root, test_mode, modality)
        self.filename_tmpl = filename_tmpl

    def load_annotations(self):
        """load annotations"""
        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                frame_dir, total_frames, label = line.split(' ')
                if self.data_root is not None:
                    frame_dir = osp.join(self.data_root, frame_dir)
                video_infos.append(
                    dict(
                        filename=frame_dir,
                        total_frames=int(total_frames),
                        label=int(label)))
        return video_infos

    def prepare_frames(self, idx):
        """prepare_frames"""
        results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        results['test_mode'] = self.test_mode
        return self.pipeline(results)
