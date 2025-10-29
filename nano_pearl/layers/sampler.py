import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass


def norm_logits(logits: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    """Normalize 2D logits [bs, vocab] with temperature.
    - temperature == 0 → one-hot on argmax
    - temperature  > 0 → softmax(logits / temperature)
    """
    if (temperature == 0.0).all():
        return F.one_hot(logits.argmax(dim=-1), num_classes=logits.size(-1)).to(logits.dtype)
    elif (temperature != 0.0).all():
        return torch.softmax(logits / temperature.unsqueeze(dim=1), dim=-1).to(logits.dtype)
    else:
        raise ValueError(f"temperature: {temperature.tolist()}. Currently temperature should be all 0 or all non-zero.")

class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        if (temperatures == 0).all():
            return self.greedy(logits, temperatures)
        elif (temperatures != 0).all():
            return self.sample(logits, temperatures)
        else:
            raise ValueError(f"temperatures: {temperatures.tolist()}. Currently temperatures should be all 0 or all non-zero.")
    
    @torch.compile
    def sample(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens

    def greedy(self, logits: torch.Tensor, temperatures: torch.Tensor):
        return logits.argmax(dim=-1)
    


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        pass
        # assert self.temperature > 1e-10, "greedy sampling is not permitted"
