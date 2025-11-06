import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nano_pearl.pearl_config import TPParams


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator

def pad_tensor(tensor: torch.Tensor, pad_size: int, dim: int = 0):
    assert tensor.dim() <= 2, "only support 1D or 2D tensor"
    assert tensor.dim() > dim, "illegal padding dimension"
    if tensor.dim() == 2:
        pad_param = (0, 0, 0, pad_size) if dim == 0 else (0, pad_size, 0, 0)
    else:
        pad_param = (0, pad_size)
    return F.pad(tensor, pad_param)

class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        tp_params: TPParams,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = tp_params.local_rank
        self.tp_size = tp_params.tp_size
        self.tp_params = tp_params
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        tp_params: TPParams,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, tp_params, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        tp_params: TPParams,
        bias: bool = False,
    ):
        tp_size = tp_params.tp_size
        super().__init__(input_size, divide(output_size, tp_size), tp_params, bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        num_pad = max(0, shard_size - (loaded_weight.size(self.tp_dim) - start_idx))
        loaded_weight = pad_tensor(loaded_weight, num_pad, self.tp_dim)
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        tp_params: TPParams,
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), tp_params, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        num_pad = max(0, param_data.size(self.tp_dim) * self.tp_size - loaded_weight.size(self.tp_dim))
        loaded_weight = pad_tensor(loaded_weight, num_pad, self.tp_dim)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        tp_params: TPParams,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = tp_params.tp_size
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, tp_params, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        num_pad = max(0, param_data.size(self.tp_dim) * self.tp_size - loaded_weight.size(self.tp_dim))
        loaded_weight = pad_tensor(loaded_weight, num_pad, self.tp_dim)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        tp_params: TPParams,
        bias: bool = False,
    ):
        tp_size = tp_params.tp_size
        super().__init__(divide(input_size, tp_size), output_size, tp_params, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        num_pad = max(0, shard_size - (loaded_weight.size(self.tp_dim) - start_idx))
        loaded_weight = pad_tensor(loaded_weight, num_pad, self.tp_dim if param.dim() == 2 else 0)
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y, group=self.tp_params.group)
        return y
