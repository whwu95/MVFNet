"""eval hooks
"""
import os
import os.path as osp

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import Hook, obj_from_dict
from torch.utils.data import Dataset

from ... import datasets
from ..parallel import collate, scatter
from .accuracy import top_k_accuracy


class DistEvalHook(Hook):
    """Distributed evaluation hook based on epochs."""

    def __init__(self, dataset, interval=1, distributed=True):
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        elif isinstance(dataset, dict):
            self.dataset = obj_from_dict(dataset, datasets,
                                         {'test_mode': True})
        else:
            raise TypeError(
                'dataset must be a Dataset object or a dict, not {}'.format(
                    type(dataset)))
        self.interval = interval
        self.dist = distributed

    def after_train_epoch(self, runner):
        """Called after every training epoch to evaluate the results."""

        if not self.every_n_epochs(runner, self.interval):
            return
        runner.model.eval()
        results = [None for _ in range(len(self.dataset))]
        if runner.rank == 0:
            prog_bar = mmcv.ProgressBar(len(self.dataset))
        for idx in range(runner.rank, len(self.dataset), runner.world_size):
            data = self.dataset[idx]
            data_gpu = scatter(
                collate([data], samples_per_gpu=1),
                [torch.cuda.current_device()])[0]

            # compute output
            with torch.no_grad():
                result = runner.model(return_loss=False, **data_gpu)
            results[idx] = result

            batch_size = runner.world_size
            if runner.rank == 0:
                for _ in range(batch_size):
                    prog_bar.update()

        tmp_file = osp.join(runner.work_dir, 'temp_{}.pkl'.format(runner.rank))
        mmcv.dump(results, tmp_file)
        dist.barrier()

        if runner.rank == 0:
            print('\n')
            for i in range(1, runner.world_size):
                tmp_file = osp.join(runner.work_dir, 'temp_{}.pkl'.format(i))
                tmp_results = mmcv.load(tmp_file)
                for idx in range(i, len(results), runner.world_size):
                    results[idx] = tmp_results[idx]
                os.remove(tmp_file)
            self.evaluate(runner, results)
            os.remove(osp.join(runner.work_dir, 'temp_0.pkl'))

        return

    def evaluate(self, runner, results):
        """Evaluate the results."""
        raise NotImplementedError


class DistEvalTopKAccuracyHook(DistEvalHook):
    """Distributed TopK evaluation hook """

    def __init__(self, dataset, interval=1, k=(1, ), dist=True):
        super(DistEvalTopKAccuracyHook, self).__init__(dataset, interval, dist)
        self.k = k

    def evaluate(self, runner, results):
        gt_labels = []
        for i in range(len(self.dataset)):
            ann = self.dataset.video_infos[i]
            gt_labels.append(ann['label'])

        results = [res.squeeze() for res in results]
        top1, top5 = top_k_accuracy(results, gt_labels, k=self.k)
        runner.mode = 'val'
        runner.log_buffer.output['top1 acc'] = top1
        runner.log_buffer.output['top5 acc'] = top5
        runner.log_buffer.ready = True
