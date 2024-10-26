import logging
import os
from pathlib import Path
from typing import List, Tuple
import rootutils

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

# Import your model class
#from src.models.dogbreed_classifier import DogBreedClassifier
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = logging.getLogger(__name__)

def setup_logger(log_file: Path):
    logging.basicConfig(level=logging.INFO, 
                        format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

def infer(model: torch.nn.Module, image_tensor: torch.Tensor, class_names: List[str]) -> Tuple[str, float]:
    outputs = model(image_tensor.unsqueeze(0))
    _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()], outputs[0][predicted.item()].item()

@hydra.main(version_base="1.3", config_path="../configs", config_name="infer")
def main(cfg: DictConfig) -> None:
    print(f"Current working directory: {os.getcwd()}")
    print(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Create log directory if it doesn't exist
    log_dir = Path(cfg.paths.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    setup_logger(log_dir / "infer_log.log")

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: torch.nn.Module = hydra.utils.instantiate(cfg.model)

    log.info(f"Loading model checkpoint: {cfg.ckpt_path}")
    checkpoint = torch.load(cfg.ckpt_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    log.info(f"Input folder: {cfg.input_folder}")
    log.info(f"Output folder: {cfg.output_folder}")

    log.info("Starting inference...")
    # Your inference logic here
    # process_images(cfg, model)

    log.info("Inference completed.")

if __name__ == "__main__":
    main()
