import os
from pathlib import Path
import logging

import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L
import torch
from torch.utils.data import DataLoader

import rootutils

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Imports that require root directory setup
from src.utils.logging_utils import setup_logger, task_wrapper

log = logging.getLogger(__name__)

@task_wrapper
def evaluate(
    cfg: DictConfig,
    model: L.LightningModule,
    datamodule: L.LightningDataModule,
):
    log.info("Starting evaluation!")
    
    # Prepare the data
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()

    # Prepare the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Evaluation loop
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    log.info(f"Test Accuracy: {accuracy:.2f}%")

    return {"accuracy": accuracy}

@hydra.main(version_base="1.3", config_path="../configs", config_name="evaluate")
def main(cfg: DictConfig):
    # Set up paths
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    log_dir = Path(cfg.paths.log_dir)

    # Set up logger
    setup_logger(log_dir / "evaluate_log.log")

    # Initialize DataModule
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)

    # Initialize Model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: L.LightningModule = hydra.utils.instantiate(cfg.model)

    # Load model weights
    if cfg.get("ckpt_path"):
        log.info(f"Loading checkpoint: {cfg.ckpt_path}")
        model = model.load_from_checkpoint(cfg.ckpt_path)
    else:
        log.warning("No checkpoint path provided. Using initial model weights.")

    # Run evaluation
    evaluate(cfg, model, datamodule)

if __name__ == "__main__":
    main()
