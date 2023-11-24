import config.fine_tuning_config as ft_config
import torch
import os
import torch.distributed as dist
import random
import utils.train_utils as train_utils


def fine_tuning_initialization():
    train_config = ft_config.TrainCfg()
    fsdp_config = ft_config.FSDPConfig()
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)
    return train_config, fsdp_config


def fsdp_initialization(config: ft_config.TrainCfg):
    if config.enable_fsdp:
        dist.init_process_group('nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
    else:
        rank, local_rank = None, None
    # world_size = int(os.environ["WORLD_SIZE"])
    if dist.is_initialized() and local_rank is not None:
        torch.cuda.set_device(local_rank)
        train_utils.clear_gpu_cache(local_rank)
        train_utils.setup_environ_flags(rank)
    else:
        pass
    return local_rank, rank

# def get_ft_mode(config: ft_config.TrainCfg):
#     mode = None
#     if config.enable_fsdp and config.low_cpu_fsdp:
#         mode = 'low_cpu_fsdp'
#     elif config.enable_fsdp:
#         mode = 'fsdp'
#     if config.use_peft:
#         mode = f'peft_{mode}'
#     return mode
