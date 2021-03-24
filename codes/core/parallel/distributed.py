"""ddp"""
import torch
import torch.distributed as dist
import torch.nn as nn
from torch._utils import (_flatten_dense_tensors, _take_tensors,
                          _unflatten_dense_tensors)

from .scatter_gather import scatter_kwargs


class MMDistributedDataParallel(nn.Module):
    """MMDDP"""
    def __init__(self,
                 module,
                 dim=0,
                 broadcast_buffers=True,
                 bucket_cap_mb=25):
        super(MMDistributedDataParallel, self).__init__()
        self.is_cuda = all(
            [p.device.type == 'cuda' for p in module.parameters()])

        self.module = module
        self.dim = dim
        self.broadcast_buffers = broadcast_buffers

        self.broadcast_bucket_size = bucket_cap_mb * 1024 * 1024

        # passing a handle to torch.nn.SyncBatchNorm layer
        # self._passing_sync_batchnorm_handle([self.module])
        self._sync_params()

    def _dist_broadcast_coalesced(self, tensors, buffer_size):
        for tensors in _take_tensors(tensors, buffer_size):
            flat_tensors = _flatten_dense_tensors(tensors)
            dist.broadcast(flat_tensors, 0)
            for tensor, synced in zip(
                    tensors, _unflatten_dense_tensors(flat_tensors, tensors)):
                tensor.copy_(synced)

    def _sync_params(self):
        module_states = list(self.module.state_dict().values())
        if len(module_states) > 0:
            self._dist_broadcast_coalesced(module_states,
                                           self.broadcast_bucket_size)
        if self.broadcast_buffers:
            if torch.__version__ < '1.0':
                buffers = [b.data for b in self.module._all_buffers()]
            else:
                buffers = [b.data for b in self.module.buffers()]
            if len(buffers) > 0:
                self._dist_broadcast_coalesced(buffers,
                                               self.broadcast_bucket_size)

    def scatter(self, inputs, kwargs, device_ids):
        """scatter"""
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def forward(self, *inputs, **kwargs):
        """len(device_ids) is 1"""
        inputs, kwargs = self.scatter(inputs, kwargs,
                                      [torch.cuda.current_device()])
        return self.module(*inputs[0], **kwargs[0])

    # def _passing_sync_batchnorm_handle(self, module_copies):
    #     for dev_idx, module in enumerate(module_copies):
    #         for layer in module.modules():
    #             if isinstance(layer, torch.nn.modules.SyncBatchNorm):
    #                 assert self.is_cuda, "SyncBatchNorm layers only work with CUDA modules"
    #                 layer._specify_ddp_gpu_num(1)
