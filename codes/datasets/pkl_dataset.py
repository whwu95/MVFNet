"""pkl dataset"""
import os.path as osp

from .base import BaseDataset
from .builder import DATASETS


@DATASETS.register_module
class PklDataset(BaseDataset):
    """Pkl dataset for action recognition.

    The dataset loads raw video pkls and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    a sample video with the filepath and label, which are split with a
    whitespace. Example of a annotation file:

    ```
    some/path/000.pkl 300 1
    some/path/001.pkl 300 1
    some/path/002.pkl 300 2
    some/path/003.pkl 300 2
    some/path/004.pkl 300 3
    some/path/005.pkl 300 3
    ```
    """

    def load_annotations(self):
        """load_annotations"""
        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                filename, total_frames, label = line.split(' ')
                if self.data_root is not None:
                    filename = osp.join(self.data_root, filename)
                video_infos.append(
                    dict(
                        filename=filename,
                        total_frames=int(total_frames),
                        label=int(label)))
        return video_infos