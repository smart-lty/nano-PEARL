from dataclasses import dataclass
import torch
from nano_pearl.pearl_config import TPParams


@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None

_DRAFT_CONTEXT = Context()
_TARGET_CONTEXT = Context()

def get_context(tp_params: TPParams):
    if tp_params.is_draft:
        return _DRAFT_CONTEXT
    else:
        return _TARGET_CONTEXT

def set_context(tp_params: TPParams, is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    global _DRAFT_CONTEXT
    global _TARGET_CONTEXT
    if tp_params.is_draft:
        _DRAFT_CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)
    else:
        _TARGET_CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context(tp_params: TPParams):
    global _DRAFT_CONTEXT
    global _TARGET_CONTEXT
    if tp_params.is_draft:
        _DRAFT_CONTEXT = Context()
    else:
        _TARGET_CONTEXT = Context()