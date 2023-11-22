from dataclasses import dataclass


@dataclass()
class TrainCfg:
    model_name: str = 'user-defined'
    run_validation: bool = True
    batch_size_training: int = 4
    batching_strategy: str = "packing"
    context_length: int = 4096
    gradient_accumulation_steps: int = 1
    num_epochs: int = 3
    num_works: int = 1
    lr: float = 1e-4
    weight_decay: float = 0.0
    gamma: float = 0.85
    seed: int = 42
    use_fp16: bool = False
    mixed_precision: bool = True
    val_batch_size: int = 1
    dataset = "custom_dataset"
    peft_method: str = "lora"
    use_peft: bool = False
    output_dir: str = "user-defined"
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: bool = False
    one_gpu: bool = False
    save_model: bool = True
    dist_checkpoint_root_folder: str = "user-defined"
    dist_checkpoint_folder: str = "fine-tuned"
    save_optimizer: bool = False
    use_fast_kernels: bool = False
