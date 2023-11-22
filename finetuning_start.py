import fire
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import peft
import random
from torch import optim
from config.load_custom_datasets import get_datasets
import config.train_config as tc
import utils.config_generate_utils as cfg_tools
from utils.train_utils import train
from transformers import LlamaForCausalLM, LlamaTokenizer
from data.concatenator import ConcatDataset


def start_train(**kwargs):
    train_config = tc.TrainCfg()
    cfg_tools.update_config(train_config)
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)

    tokenizer = LlamaTokenizer.from_pretrained(train_config.model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = LlamaForCausalLM.from_pretrained(train_config.model_name,
                                             load_in_8bit=True if train_config.quantization else False,
                                             device_map="auto")
    cfg_tools.print_model_size(model, train_config)

    if train_config.quantization:
        model = peft.prepare_model_for_kbit_training(model)

    if train_config.use_peft:
        peft_config = cfg_tools.generate_peft_config(train_config, kwargs)
        model = peft.get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    model.to("cuda")

    dataset_config = cfg_tools.generate_dataset_config(train_config, kwargs)

    # Load and preprocess the dataset for training and validation
    dataset_train = get_datasets(dataset_config, tokenizer, dataset_config.train_split)

    print(f"--> Training Set Length = {len(dataset_train)}")

    dataset_test = get_datasets(dataset_config, tokenizer, split=dataset_config.test_split)

    print(f"--> Validation Set Length = {len(dataset_test)}")

    if train_config.batching_strategy == "packing":
        dataset_train = ConcatDataset(dataset_train, chunk_size=train_config.context_length)

    train_dl_kwargs = cfg_tools.get_dataloader_kwargs(train_config, dataset_train, tokenizer, "train")

    train_dataloader = DataLoader(dataset_train, num_workers=train_config.num_works, pin_memory=True, **train_dl_kwargs)
    test_dataloader = None
    if train_config.run_validation:
        if train_config.batching_strategy == "packing":
            dataset_test = ConcatDataset(dataset_test, chunk_size=train_config.context_length)
        val_dl_kwargs = cfg_tools.get_dataloader_kwargs(train_config, dataset_test, tokenizer, "test")
        test_dataloader = DataLoader(dataset_test, num_workers=train_config.num_works, pin_memory=True, **val_dl_kwargs)

    optimizer = optim.AdamW(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

    results = train(model, train_dataloader, test_dataloader, tokenizer, optimizer, scheduler,
                    train_config.gradient_accumulation_steps, train_config)
    for k, v in results.items():
        print(f'Key: {k}, Value: {v}')


if __name__ == "__main__":
    fire.Fire(start_train)
