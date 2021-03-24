"""misc"""
import numpy as np
import torch

from .flops_hook import profile


def params_count(model):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    return np.sum([p.numel() for p in model.parameters()]).item()


def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (MB).
    """
    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / (1024 * 1024)


def get_flop_stats(model, input):
    """
    Compute the gflops for the current model given the config.
    Args:
        model (model): model to compute the flop counts.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        is_train (bool): if True, compute flops for training. Otherwise,
            compute flops for testing.
    Returns:
        float: the total number of gflops of the given model.
    """
    inputs = (input.unsqueeze(0).cuda(), torch.tensor([0]).cuda())
    total_flops, total_params = profile(model, inputs, verbose=False)
    return total_flops, total_params
