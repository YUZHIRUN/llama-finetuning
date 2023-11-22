from dataclasses import dataclass


@dataclass()
class custom_dataset:
    dataset: str = "custom_dataset"
    path: str = './parquet_datasets'
    train_split: str = "train"
    test_split: str = "test"
