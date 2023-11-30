import config
import data
import utils
from torch.utils.data import DataLoader


def get_datasets_loader(train_config: config.TrainCfg, tokenizer, rank=0, **kwargs):
    dataset_config = config.custom_dataset()
    utils.update_config(dataset_config, **kwargs)
    train_dataset = config.get_datasets(dataset_config, tokenizer, split='train')
    if train_config.enable_fsdp or rank == 0:
        print(f"--> Training Set Length = {len(train_dataset)}")
    if train_config.run_validation:
        test_dataset = config.get_datasets(dataset_config, tokenizer, split='test')
        if not train_config.enable_fsdp or rank == 0:
            print(f"--> Training Set Length = {len(test_dataset)}")
    else:
        test_dataset = None
    if train_config.batching_strategy == 'packing':
        train_dataset = data.ConcatDataset(train_dataset, chunk_size=train_config.context_length)
    train_dl_kwargs = utils.get_dataloader_kwargs(train_config, train_dataset, tokenizer, mode='train')
    train_dataset_loader = DataLoader(train_dataset, num_workers=train_config.num_works, pin_memory=True,
                                      **train_dl_kwargs)
    if train_config.run_validation:
        test_dl_kwargs = utils.get_dataloader_kwargs(train_config, test_dataset, tokenizer, mode='test')
        test_dataset_loader = DataLoader(test_dataset, num_workers=train_config.num_works, pin_memory=True,
                                         **test_dl_kwargs)
    else:
        test_dataset_loader = None
    return train_dataset_loader, test_dataset_loader
