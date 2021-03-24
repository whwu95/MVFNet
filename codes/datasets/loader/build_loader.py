""" https://github.com/pytorch/pytorch/issues/973"""
import resource

from torch.distributed import get_rank, get_world_size
# from mmcv.parallel import collate
from torch.utils.data import DataLoader  # , DistributedSampler

from .sampler import DistributedSampler

# from functools import partial

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def build_dataloader(dataset,
                     videos_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     pin_memory=True,
                     **kwargs):
    """build dataloader"""
    if dist:
        rank = get_rank()
        world_size = get_world_size()
        sampler = DistributedSampler(
            dataset, world_size, rank, shuffle=shuffle)
        shuffle = False
        batch_size = videos_per_gpu
        num_workers = workers_per_gpu
    else:
        # if not kwargs.get('shuffle', True):
        #     sampler = None
        # else:
        #     sampler = GroupSampler(dataset, videos_per_gpu)
        sampler = None
        batch_size = num_gpus * videos_per_gpu
        num_workers = num_gpus * workers_per_gpu

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        # collate_fn=partial(collate, samples_per_gpu=videos_per_gpu),
        pin_memory=pin_memory,
        shuffle=shuffle,
        **kwargs)

    return data_loader
