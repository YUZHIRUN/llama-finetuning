from dataclasses import dataclass


@dataclass()
class custom_dataset:
    dataset: str = "custom_dataset"
    path: str = 'user defined path'
    train_split: str = "train"
    test_split: str = "test"
