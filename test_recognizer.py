"""test recognizer"""
import argparse
import warnings

import mmcv
import numpy as np
import torch
from mmcv.runner import obj_from_dict
from torch.nn.parallel import DataParallel, DistributedDataParallel

# from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from codes import datasets
from codes.core import (get_dist_info, init_dist, mean_class_accuracy,
                        multi_gpu_test, single_gpu_test, top_k_accuracy)
from codes.datasets import build_dataloader
from codes.models import build_recognizer
from codes.utils import load_checkpoint

# from codes.core import MMDataParallel, MMDistributedDataParallel

warnings.filterwarnings("ignore", category=UserWarning)
args = None


def parse_args():
    """parse_args"""
    parser = argparse.ArgumentParser(description='Test an action recognizer')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'mpi', 'slurm'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--average-clips',
        choices=['score', 'prob'],
        default='prob',
        help='average type when averaging test clips')
    parser.add_argument('--out', help='output result file',
                        default='default.pkl')
    # only for TSN3D
    parser.add_argument('--fcn_testing', action='store_true',
                        help='use fcn testing for 3D convnet')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args


def main():
    """main"""
    global args
    args = parse_args()

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    cfg.gpus = args.gpus
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    # pass arg of fcn testing
    if args.fcn_testing:
        cfg.model.update({'fcn_testing': True})
        cfg.model['cls_head'].update({'fcn_testing': True})

    if cfg.test_cfg is None:
        cfg.test_cfg = dict(average_clips=args.average_clips)
    else:
        cfg.test_cfg.average_clips = args.average_clips

    # for regular testing
    # pipeline_type = [op['type'] for op in cfg.test_pipeline]
    # if 'ThreeCrop' in pipeline_type:
    #     cfg.model.cls_head.spatial_size = 8

    dataset = obj_from_dict(cfg.data.test, datasets, dict(test_mode=True))

    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    model = build_recognizer(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    data_loader = build_dataloader(
        dataset,
        num_gpus=1 if distributed else cfg.gpus,
        videos_per_gpu=1,
        workers_per_gpu=1,
        dist=distributed,
        shuffle=False)

    if distributed:
        # model = MMDistributedDataParallel(model.cuda())
        model = DistributedDataParallel(model.cuda(), device_ids=[
                                        torch.cuda.current_device()])
        outputs = multi_gpu_test(model, data_loader)
        rank, _ = get_dist_info()
    else:
        # model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()
        model = DataParallel(model, device_ids=range(cfg.gpus)).cuda()
        outputs = single_gpu_test(model, data_loader)
        rank = 0

    if args.out and rank == 0:
        # print('\nwriting results to {}'.format(args.out))
        # for videos_per_gpu > 1, vstack list of array
        results = np.vstack(outputs)  # list(1 x n_class) -> n_video x n_class
        # outputs = np.vstack(outputs) [:, np.newaxis, :]
        mmcv.dump(results, args.out)

        gt_labels = []
        for i in range(len(dataset)):
            ann = dataset.video_infos[i]
            gt_labels.append(ann['label'])

        # if args.use_softmax:
        #     print("Averaging score over {} clips with softmax".format(
        #         outputs[0].shape[0]))
        #     results = [softmax(res, dim=1).mean(axis=0) for res in outputs]
        # else:
        #     print("Averaging score over {} clips without softmax (ie, raw)".
        #           format(outputs[0].shape[0]))
        #     results = [res.mean(axis=0) for res in outputs]
        # results = [res.squeeze() for res in outputs]
        top1, top5 = top_k_accuracy(results, gt_labels, k=(1, 5))
        mean_acc = mean_class_accuracy(results, gt_labels)
        print("Mean Class Accuracy = {:.02f}".format(mean_acc * 100))
        print("Top-1 Accuracy = {:.02f}".format(top1 * 100))
        print("Top-5 Accuracy = {:.02f}".format(top5 * 100))


if __name__ == '__main__':
    main()
