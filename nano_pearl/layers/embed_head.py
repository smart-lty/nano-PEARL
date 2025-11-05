import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nano_pearl.utils.context import get_context
from nano_pearl.pearl_config import TPParams
from nano_pearl.layers.linear import pad_tensor


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        tp_params: TPParams,
    ):
        super().__init__()
        self.tp_params = tp_params
        self.local_tp_rank = tp_params.local_rank
        self.tp_size = tp_params.tp_size
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.local_tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.local_tp_rank * shard_size
        num_pad = max(0, shard_size - (loaded_weight.size(0) - start_idx))
        loaded_weight = pad_tensor(loaded_weight, num_pad, 0)
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y, group=self.tp_params.group)
        return y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        tp_params: TPParams,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim, tp_params)
        self.valid_vocab_size = getattr(tp_params, "valid_vocab_size", num_embeddings)

    def forward(self, x: torch.Tensor):
        context = get_context(self.tp_params)
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.local_tp_rank == 0 else None
            dist.gather(logits, all_logits, self.tp_params.master_rank, group=self.tp_params.group)
            logits = torch.cat(all_logits, -1) if self.local_tp_rank == 0 else None
            logits = logits[..., :self.valid_vocab_size] if self.local_tp_rank == 0 else None
        return logits
