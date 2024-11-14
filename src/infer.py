import logging
import os
from pathlib import Path
from typing import List, Tuple
import rootutils

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

# Import your model class
from src.models.dogbreed_classifier import DogBreedClassifier

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
log = logging.getLogger(__name__)

# Function to set up logging
def setup_logger(log_file: Path):
    logging.basicConfig(level=logging.INFO, 
                        format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

# Inference function
def infer(model: pl.LightningModule, image_tensor: torch.Tensor, class_names: List[str]) -> Tuple[str, float]:
    model.eval()
    with torch.no_grad():
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
    model = DogBreedClassifier.load_from_checkpoint(cfg.ckpt_path, **cfg.model, strict=False)

    log.info(f"Loading model checkpoint from: {cfg.ckpt_path}")
    print(f"Resolved Checkpoint Path: {cfg.ckpt_path}")

    # Ensure the model is in evaluation mode
    model.eval()

    log.info(f"Input folder: {cfg.input_folder}")
    log.info(f"Output folder: {cfg.output_folder}")

    log.info("Starting inference...")
    # Run inference
    process_images(cfg, model)

    log.info("Inference completed.")

# Function to process images for inference
def process_images(cfg: DictConfig, model: pl.LightningModule):
    input_folder = Path(cfg.input_folder)
    output_folder = Path(cfg.output_folder)
    class_names = cfg.class_names

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load and preprocess images
    for image_path in input_folder.glob("*.jpg"):
        log.info(f"Processing image: {image_path.name}")
        image_tensor = preprocess_image(image_path)

        # Run inference
        predicted_class, confidence = infer(model, image_tensor, class_names)
        log.info(f"Predicted: {predicted_class} with confidence {confidence:.2f}")

        # Save the result
        result_path = output_folder / f"{image_path.stem}_result.txt"
        with open(result_path, "w") as f:
            f.write(f"Predicted: {predicted_class}\nConfidence: {confidence:.2f}")

@rank_zero_only
def preprocess_image(image_path: Path) -> torch.Tensor:
    from PIL import Image
    import torchvision.transforms as T

    # Define preprocessing transformations
    transform = T.Compose([
        T.Resize((224, 224)),    # Resize image
        T.ToTensor(),            # Convert to tensor
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image)

    return image_tensor

if __name__ == "__main__":
    main()
