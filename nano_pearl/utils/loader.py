import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open
from tqdm import tqdm

from nano_pearl.utils.pearl_logger import logger


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param_data = param.data
    param_data.zero_()
    common_shape = tuple(min(a, b) for a, b in zip(param_data.shape, loaded_weight.shape))
    slices = tuple(slice(0, dim) for dim in common_shape)
    param_data[slices].copy_(loaded_weight[slices])


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    files = sorted(glob(os.path.join(path, "*.safetensors")))
    
    is_master = model.lm_head.local_tp_rank == 0
    pbar = tqdm(files, dynamic_ncols=True, desc="Loading model", disable=not is_master)
    
    for file in pbar:
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
