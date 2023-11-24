from transformers import LlamaForCausalLM, LlamaConfig, LlamaTokenizer
from config import fine_tuning_config
from utils import config_generate_utils as config_utils
import torch
import peft


def load_llama_integrate(config: fine_tuning_config.TrainCfg):
    use_cache = False if config.enable_fsdp else None
    is_load_8bit = True if config.quantization else None
    is_device_auto = 'auto' if config.quantization else None
    model = LlamaForCausalLM.from_pretrained(config.model_name, load_in_8bit=is_load_8bit, device_map=is_device_auto,
                                             use_cache=use_cache)
    return model


def load_llama_from_config(config: fine_tuning_config.TrainCfg):
    use_cache = False if config.enable_fsdp else None
    llama_config = LlamaConfig.from_pretrained(config.model_name)
    llama_config.use_cache = use_cache
    with torch.device('meta'):
        model = LlamaForCausalLM(llama_config)
    return model


def load_llama_fast_kernels(model, config: fine_tuning_config.TrainCfg):
    if config.enable_fsdp and config.use_fast_kernels:
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model)
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")
    else:
        pass
    return model


def load_llama_model(config: fine_tuning_config.TrainCfg, rank):
    if config.enable_fsdp and config.low_cpu_fsdp:
        if rank == 0:
            model = load_llama_integrate(config)
        else:
            model = load_llama_from_config(config)
    else:
        model = load_llama_integrate(config)
    model = load_llama_fast_kernels(model, config)
    return model


def load_llama_mode_use_peft(model, config: fine_tuning_config.TrainCfg):
    if config.use_peft:
        peft_config = config_utils.generate_peft_config(config)
        model_peft = peft.get_peft_model(model, peft_config)
        model_peft.print_trainable_parameters()
    else:
        model_peft = model
    return model_peft


def load_llama_tokenizer(config: fine_tuning_config.TrainCfg):
    tokenizer = LlamaTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer
