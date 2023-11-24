from config import fine_tuning_config as ft_config
from config import peft_info
from config import datasets_info
from dataclasses import asdict
import torch.distributed as dist
from torch.utils.data import DistributedSampler
from peft import LoraConfig, AdaptionPromptConfig, PrefixTuningConfig
from transformers import default_data_collator
from transformers.data import DataCollatorForSeq2Seq
from data.sampler import LengthBasedBatchSampler, DistributedLengthBasedBatchSampler


def update_config(config, **kwargs):
    if isinstance(config, (tuple, list)):
        for c in config:
            update_config(c, **kwargs)
    else:
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
            elif "." in k:
                # allow --some_config.some_param=True
                config_name, param_name = k.split(".")
                if type(config).__name__ == config_name:
                    if hasattr(config, param_name):
                        setattr(config, param_name, v)
                    else:
                        # In case of specialized config we can warm user
                        print(f"Warning: {config_name} does not accept parameter: {k}")
            elif isinstance(config, ft_config.TrainCfg):
                print(f"Warning: unknown parameter {k}")


def generate_peft_config(train_config):
    configs = (peft_info.lora_config, peft_info.llama_adapter_config, peft_info.prefix_config)
    peft_configs = (LoraConfig, AdaptionPromptConfig, PrefixTuningConfig)
    names = tuple(c.__name__.rstrip("_config") for c in configs)
    assert train_config.peft_method in names, f"Peft config not found: {train_config.peft_method}"
    config = configs[names.index(train_config.peft_method)]()
    params = asdict(config)
    peft_config = peft_configs[names.index(train_config.peft_method)](**params)

    return peft_config


def generate_dataset_config(train_config, kwargs):
    dataset_config = datasets_info.custom_dataset()

    return dataset_config


def get_dataloader_kwargs(train_config, dataset, tokenizer, mode):
    kwargs = {}
    batch_size = train_config.batch_size_training if mode == "train" else train_config.val_batch_size
    if train_config.batching_strategy == "padding":
        if train_config.enable_fsdp:
            kwargs["batch_sampler"] = DistributedLengthBasedBatchSampler(
                dataset,
                batch_size=batch_size,
                rank=dist.get_rank(),
                num_replicas=dist.get_world_size(),
                shuffle=mode == "train",
            )
        else:
            kwargs["batch_sampler"] = LengthBasedBatchSampler(dataset, batch_size, drop_last=True,
                                                              shuffle=mode == "train")
        kwargs["collate_fn"] = DataCollatorForSeq2Seq(tokenizer)
    elif train_config.batching_strategy == "packing":
        if train_config.enable_fsdp:
            kwargs["sampler"] = DistributedSampler(
                dataset,
                rank=dist.get_rank(),
                num_replicas=dist.get_world_size(),
                shuffle=mode == "train",
            )
        kwargs["batch_size"] = batch_size
        kwargs["drop_last"] = True
        kwargs["collate_fn"] = default_data_collator
    else:
        raise ValueError(f"Unknown batching strategy: {train_config.batching_strategy}")

    return kwargs


def print_model_size(model, config, rank: int = 0) -> None:
    if rank == 0:
        print(f"--> Model {config.model_name}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> {config.model_name} has {total_params / 1e6} Million params\n")
