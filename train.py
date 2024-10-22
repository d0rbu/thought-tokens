import arguably
import importlib

from typing import Any
from core.logger import logger
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoModelForCausalLM, AutoTokenizer
from pytorch_lightning import LightningModule, LightningDataModule, Trainer


@arguably.command
def train(
    *args: Any,
    script: str,
    data: str = "minipile",
    model: str = "meta-llama/Llama-3.2-1B",
    batch_size: int = 1,
    init_size: int = 42,
    context_length: int = 1024,  # sorry i am gpu poor :(
    warmup_steps: int = 1000,
    lr: float = 5e-5,
    grad_accum: int = 32,
    num_thought_tokens: int = 1024,
    max_epochs: int = 1,
    num_workers: int = 16,
    val_every: int = 100,
) -> None:
    script_path = f"train.{script}"
    script_module = importlib.import_module(script_path)
    lightning_modules = [(name, cls) for name, cls in script_module.__dict__.items() if isinstance(cls, type) and issubclass(cls, LightningModule)]
    assert len(lightning_modules) == 1, f"Expected exactly one LightningModule in {script_path}, found {len(lightning_modules)}: {lightning_modules}"

    lightning_module_name, lightning_module = lightning_modules[0]

    data_path = f"data.{data}"
    data_module = importlib.import_module(data_path)
    lightning_data_modules = [(name, cls) for name, cls in data_module.__dict__.items() if isinstance(cls, type) and issubclass(cls, LightningDataModule)]
    assert len(lightning_data_modules) == 1, f"Expected exactly one LightningDataModule in {data_path}, found {len(lightning_data_modules)}: {lightning_data_modules}"

    lightning_data_module_name, lightning_data_module = lightning_data_modules[0]

    model_str = model
    model = AutoModelForCausalLM.from_pretrained(model_str)
    tokenizer = AutoTokenizer.from_pretrained(model_str)

    tokenizer.model_max_length = context_length

    logger.info(f"Training {model_str} with {lightning_module_name} on {lightning_data_module_name}.")
    logger.info(f"Using {num_thought_tokens} thought tokens.")

    wandb_logger = WandbLogger(project="thought-token-lm")

    module = lightning_module(model=model, tokenizer=tokenizer, thought_token_embeddings=num_thought_tokens, warmup_steps=warmup_steps, lr=lr)
    data_module = lightning_data_module(batch_size=batch_size, init_size=init_size, num_workers=num_workers)
    trainer = Trainer(max_epochs=max_epochs, accumulate_grad_batches=grad_accum, val_check_interval=(val_every * grad_accum) // (batch_size), logger=wandb_logger)

    wandb_logger.log_hyperparams(module.hparams)

    trainer.fit(module, datamodule=data_module)


if __name__ == "__main__":
    arguably.run()
