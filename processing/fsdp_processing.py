import utils.train_utils as train_utils
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from config.fine_tuning_config import TrainCfg, FSDPConfig
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
import torch
import policies


def fsdp_auto_wrap_policy(model, transformer_layer_name):
    import functools

    from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy

    from peft.tuners import PrefixEncoder, PromptEmbedding, PromptEncoder

    def lambda_policy_fn(module):
        if (
                len(list(module.named_children())) == 0
                and getattr(module, "weight", None) is not None
                and module.weight.requires_grad
        ):
            return True
        return False

    lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=(
            PrefixEncoder,
            PromptEncoder,
            PromptEmbedding,
            transformer_layer_name,
            # FullyShardedDataParallelPlugin.get_module_class_from_name(
            #     model, transformer_layer_name
            # ),
        ),
    )
    auto_wrap_policy = functools.partial(_or_policy, policies=[lambda_policy, transformer_wrap_policy])
    return auto_wrap_policy


def fsdp_wrap(model, configs: (TrainCfg, FSDPConfig), rank):
    train_config = configs[0]
    fsdp_config = configs[1]
    if train_config.enable_fsdp:
        if not train_config.use_peft and train_config.freeze_layers:
            train_utils.freeze_transformer_layers(model, train_config.num_freeze_layers)
        mixed_precision_policy, wrapping_policy = train_utils.get_policies(fsdp_config, rank)
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer)
        model = FSDP(
            model,
            auto_wrap_policy=my_auto_wrapping_policy if train_config.use_peft else wrapping_policy,
            cpu_offload=CPUOffload(offload_params=True) if fsdp_config.fsdp_cpu_offload else None,
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            sync_module_states=train_config.low_cpu_fsdp,
            param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
            if train_config.low_cpu_fsdp and rank != 0 else None,
        )
        if fsdp_config.fsdp_activation_checkpointing:
            policies.apply_fsdp_checkpointing(model)
    elif not train_config.quantization and not train_config.enable_fsdp:
        model.to('cuda')
    return model
