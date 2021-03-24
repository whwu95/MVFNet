"""count flops"""
import argparse

import mmcv
import torch
from mmcv.runner import obj_from_dict

from codes import datasets
from codes.models import build_recognizer
from codes.utils import get_flop_stats


def parse_args():
    """parse"""
    parser = argparse.ArgumentParser(description='Test an action recognizer')
    parser.add_argument('config', help='test config file path')
    args = parser.parse_args()
    return args


def main():
    """main"""
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    dataset = obj_from_dict(cfg.data.val, datasets, dict(test_mode=True))

    # get flops and params
    flop_input = dataset[0]['img_group']
    flop_model = build_recognizer(
        cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    # print(flop_model)

    flops, params = get_flop_stats(flop_model.cuda(), flop_input)

    # import numpy as np
    # params = np.sum([p.numel() for p in flop_model.parameters()]).item()
    # from fvcore.nn.flop_count import flop_count
    # inputs = (flop_input.unsqueeze(0).cuda(), torch.LongTensor([1]).cuda())
    # gflop_dict, _ = flop_count(flop_model, inputs)
    # flops = sum(gflop_dict.values())

    del(flop_model)
    print(" GFLOPs: %.3f | Params: %.2f" %
          (round(flops / 10. ** 9, 3), (round(params / 10 ** 6, 2))))


if __name__ == '__main__':
    main()
