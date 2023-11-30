import fire
from torch.optim.lr_scheduler import StepLR
import peft
from policies import *
from torch import optim
from utils import *
from processing import *


def start_train(**kwargs):
    train_config, fsdp_config = fine_tuning_initialization()
    update_config((train_config, fsdp_config), **kwargs)
    local_rank, rank = fsdp_initialization(train_config)

    model = load_llama_model(train_config, rank)
    tokenizer = load_llama_tokenizer(train_config)
    print_model_size(model, train_config, rank)
    if train_config.quantization:
        model = peft.prepare_model_for_kbit_training(model)
    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)
    model = load_llama_mode_use_peft(model, train_config)
    model = fsdp_wrap(model, (train_config, fsdp_config), rank)
    train_dataset_loader, test_dataset_loader = get_datasets_loader(train_config, tokenizer, rank, **kwargs)
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == 'anyprecision':
        optimizer = AnyPrecisionAdamW(
            model.parameters(),
            lr=train_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
            weight_decay=train_config.weight_decay,
        )
    else:
        optimizer = optim.AdamW(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

    results = TRAIN(model, train_dataset_loader, test_dataset_loader, tokenizer, optimizer, scheduler,
                    train_config.gradient_accumulation_steps, train_config, fsdp_config, local_rank, rank)
    if not train_config.enable_fsdp or rank == 0:
        for k, v in results.items():
            print(f'Key: {k}, Value: {v}')


if __name__ == "__main__":
    fire.Fire(start_train)
