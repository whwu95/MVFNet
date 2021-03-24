"""data parallel"""
from torch.nn.parallel import DataParallel

from .scatter_gather import scatter_kwargs


class MMDataParallel(DataParallel):
    """[summary]

    Args:
        DataParallel ([type]): [description]
    """
    def scatter(self, inputs, kwargs, device_ids):
        """scatter"""
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)
