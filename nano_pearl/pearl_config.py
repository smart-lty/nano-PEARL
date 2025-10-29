import os
from nano_pearl.utils.pearl_logger import logger, get_model_name
from dataclasses import dataclass
from transformers import AutoConfig
import torch.distributed as dist


@dataclass
class TPParams:
    rank: int
    group: dist.ProcessGroup
    group_name: str
    local_rank: int
    master_rank: int
    is_draft: bool
    tp_size: int


class BaseConfig:
    def __init__(self, model: str, tensor_parallel_size: int, devices: list[int], group_name: str):
        self.model = model
        self.tensor_parallel_size = tensor_parallel_size
        self.devices = devices
        self.group_name = group_name
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.eos = self.hf_config.eos_token_id
        self.master_rank = self.devices[0]
        logger.info(f"Model={get_model_name(self.model)}")
        logger.info(f"TP={self.tensor_parallel_size}")
        logger.info(f"Devices={self.devices}")
        logger.info(f"GroupName={self.group_name}")
        logger.info(f"Architectures={self.hf_config.architectures[0]}")
        logger.info(f"Vocab_Size={self.hf_config.vocab_size}")
        logger.info(f"Eos={self.eos}")


@dataclass
class PEARLConfig:
    draft_model_path: str
    target_model_path: str
    draft_tensor_parallel_size: int = 2
    target_tensor_parallel_size: int = 2
    draft_group_name: str = "draft_group"
    target_group_name: str = "target_group"
    max_num_batched_tokens: int = 16384 # 8192 for 40GB GPUs
    max_num_seqs: int = 512 # 128 or 256 for 40GB GPUs
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    enforce_eager: bool = False
    gamma: int = -1
    def __post_init__(self):
        logger.info("="*50)
        logger.info(f"Loading Draft Config:")
        draft_devices = list(range(self.draft_tensor_parallel_size))
        self.draft_config = BaseConfig(self.draft_model_path, self.draft_tensor_parallel_size, draft_devices, self.draft_group_name)
        logger.info("="*50)
        logger.info(f"Loading Target Config:")
        target_devices = list(range(len(draft_devices), len(draft_devices) + self.target_tensor_parallel_size))
        self.target_config = BaseConfig(self.target_model_path, self.target_tensor_parallel_size, target_devices, self.target_group_name)
        logger.info("="*50)
        logger.info(f"Global_Config:")
        logger.info(f"Max_Num_Batched_Tokens={self.max_num_batched_tokens}")
        logger.info(f"Max_Num_Seqs={self.max_num_seqs}")
        logger.info(f"Max_Model_Len={self.max_model_len}")
        logger.info(f"GPU_Memory_Utilization={self.gpu_memory_utilization}")
        logger.info(f"Enforce_Eager={self.enforce_eager}")
        logger.info(f"Gamma (Window_Size)={self.gamma}, [-1 means auto-set]")
        assert self.draft_config.eos == self.target_config.eos
        assert (self.draft_config.tensor_parallel_size + self.target_config.tensor_parallel_size) <= 8
        assert self.max_num_batched_tokens >= self.max_model_len
        self.world_size = self.draft_config.tensor_parallel_size + self.target_config.tensor_parallel_size
        logger.info(f"World_Size={self.world_size}")
        logger.info("="*50)
    
