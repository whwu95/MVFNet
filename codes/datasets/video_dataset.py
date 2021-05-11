"""video dataset"""
import os.path as osp
import copy
from .base import BaseDataset
from .builder import DATASETS
import random
# TODO: More efficient
@DATASETS.register_module
class VideoDataset(BaseDataset):
    """Video dataset for action recognition.
    The dataset loads raw videos and apply specified transforms to return a
    dict containing the frame tensors and other information.
    The ann_file is a text file with multiple lines, and each line indicates
    a sample video with the filepath and label, which are split with a
    whitespace. Example of a annotation file:
    ```
    some/path/000.mp4 1
    some/path/001.mp4 1
    some/path/002.mp4 2
    some/path/003.mp4 2
    some/path/004.mp4 3
    some/path/005.mp4 3
    ```
    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        data_root (str): Path to a directory where videos are held.
        num_retries (int): number of retries.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root=None,
                 test_mode=False,
                 num_retries=10,
                 modality=None):
        super(VideoDataset, self).__init__(ann_file, pipeline,
                                           data_root, test_mode, modality)
        self._num_retries = num_retries

    def load_annotations(self):
        """load_annotations"""
        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                line_split = line.strip().split()
                if len(line_split) == 1:  # for extract feats
                    filename, label = line_split[0], 0
                else:
                    filename, label = line.strip().split()
                if self.data_root is not None:
                    filename = osp.join(self.data_root, filename)
                video_infos.append(dict(filename=filename, label=int(label)))
        return video_infos

    def prepare_frames(self, idx):
        """get frames"""
        for i_try in range(self._num_retries):
            results = copy.deepcopy(self.video_infos[idx])
            results['modality'] = self.modality
            results['test_mode'] = self.test_mode
            results['vid_idx'] = idx
            data = self.pipeline(results)
            if data is None:
                print("Failed to decode video idx {} from {}; trial {}".format(
                    idx, results['filename'], i_try)
                )
                idx = random.randint(0, len(self.video_infos))
                continue
            return data
        raise RuntimeError(
            "Failed to fetch video after {} retries.".format(
                self._num_retries
            )
        )
