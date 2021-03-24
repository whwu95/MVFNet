# !/usr/bin/env python3
"""base dataset"""
import copy
from abc import ABCMeta, abstractmethod

from torch.utils.data import Dataset

from .pipelines import Compose


class BaseDataset(Dataset, metaclass=ABCMeta):
    """Base class for datasets.

    All datasets to process video should subclass it.
    All subclasses should overwrite:
        Methods:`load_annotations`, supporting to load information
            from an annotation file.
        Methods:`prepare_frames`, providing data.

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        data_root (str): Path to a directory where videos are held.
    """

    def __init__(self, ann_file, pipeline, data_root=None,
                 test_mode=False, modality=None):
        super(BaseDataset, self).__init__()

        self.ann_file = ann_file
        self.data_root = data_root
        self.test_mode = test_mode
        self.pipeline = Compose(pipeline)
        self.video_infos = self.load_annotations()
        self.modality = modality

    @abstractmethod
    def load_annotations(self):
        pass

    def prepare_frames(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        results['modality'] = self.modality
        results['test_mode'] = self.test_mode
        return self.pipeline(results)

    def __len__(self):
        return len(self.video_infos)

    def __getitem__(self, idx):
        return self.prepare_frames(idx)
