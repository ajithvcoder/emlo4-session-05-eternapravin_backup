import os
from pathlib import Path
import logging

import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L
from lightning.pytorch.loggers import Logger
from typing import List
from lightning.pytorch.callbacks import ModelCheckpoint
from typing import Any, Dict

import rootutils

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Imports that require root directory setup
from src.utils.logging_utils import setup_logger, task_wrapper

log = logging.getLogger(__name__)


def instantiate_callbacks(callback_cfg: DictConfig) -> List[L.Callback]:
    callbacks: List[L.Callback] = []
    if not callback_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    for _, cb_conf in callback_cfg.items():
        if "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    loggers: List[Logger] = []
    if not logger_cfg:
        log.warning("No logger configs found! Skipping..")
        return loggers

    for _, lg_conf in logger_cfg.items():
        if "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            loggers.append(hydra.utils.instantiate(lg_conf))

    return loggers

class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, custom_dir: str = "/checkpoints/", **kwargs):
        # Ensure the custom directory exists
        os.makedirs(custom_dir, exist_ok=True)

        # Pass the custom directory and additional parameters to the base class
        super().__init__(dirpath=custom_dir, **kwargs)

    def save_checkpoint(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Override this method if you want to customize the saving logic further."""
        # You can add custom logic here if needed
        print(f"Saving checkpoint to: {self.dirpath}")
        super().save_checkpoint(trainer, pl_module)

    def format_checkpoint_name(self, epoch, step, metrics):
        """Custom formatting for checkpoint filenames."""
        # Example filename: 'custom-epoch=001-step=1000.ckpt'
        #filename = f"custom-epoch={epoch:03d}-step={step}.ckpt"
        filename = f"epoch_best"
        return filename

@task_wrapper
def train(
    cfg: DictConfig,
    trainer: L.Trainer,
    model: L.LightningModule, 
    datamodule: L.LightningDataModule,
):
    log.info("Starting training!")
    trainer.fit(model, datamodule)
    train_metrics = trainer.callback_metrics
    log.info(f"Training metrics:\n{train_metrics}")


@task_wrapper
def test(
    cfg: DictConfig,
    trainer: L.Trainer,
    model: L.LightningModule,
    datamodule: L.LightningDataModule,
):
    log.info("Starting testing!")
    if trainer.checkpoint_callback.best_model_path:
        log.info(
            f"Loading best checkpoint: {trainer.checkpoint_callback.best_model_path}"
        )
        test_metrics = trainer.test(
            model, datamodule, ckpt_path=trainer.checkpoint_callback.best_model_path
        )
    else:
        log.warning("No checkpoint found! Using current model weights.")
        test_metrics = trainer.test(model, datamodule)
    log.info(f"Test metrics:\n{test_metrics}")


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    # Set up paths
    print(OmegaConf.to_yaml(cfg))
    log_dir = Path(cfg.paths.log_dir)

    # Set up logger
    setup_logger(log_dir / "train_log.log")

    # Initialize DataModule
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)
    print("Printing of DataModule")
    print(OmegaConf.to_yaml(cfg.data))

    # Initialize Model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: L.LightningModule = hydra.utils.instantiate(cfg.model)
    print("Printing of ModelConfig")
    print(OmegaConf.to_yaml(cfg.model))

    # Set up callbacks
    callbacks: List[L.Callback] = instantiate_callbacks(cfg.get("callbacks"))
    print("Printing of CallbackConfigs")
    print(OmegaConf.to_yaml(cfg.callbacks))

    # Set up loggers
    loggers: List[Logger] = instantiate_loggers(cfg.get("logger"))
    print("Printing of LoggerConfig")
    print(OmegaConf.to_yaml(cfg.logger))
    # Initialize Trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=loggers,
    )

    # Train the model
    if cfg.get("train"):
        train(cfg, trainer, model, datamodule)

    # Test the model
    if cfg.get("test"):
        test(cfg, trainer, model, datamodule)


if __name__ == "__main__":
    main()