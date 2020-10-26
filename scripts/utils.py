import sys
import toml
from pathlib import Path
from scripts.inference_manager import InferSegmentation


def create_inference():
    base_dir_path = Path(__file__).resolve().parents[1]
    config_path =  f"{base_dir_path}/cfg/semantic_segmentation.toml"
    print(config_path)
    with open(str(config_path), "r") as f:
        config = toml.load(f)

    return InferSegmentation(
        weights=str(config["weights"]),
        architecture=config["architecture"],
        encoder=config["encoder"],
        depth=config["depth"],
        in_channels=config["in_channels"],
        classes=config["classes"],
        activation=config["activation"],
        resize=config["resize"],
        gpu=config["gpu_id"]
    )


def add_dummy_dim(image):
    return image.reshape(
        (1, image.shape[0], image.shape[1], image.shape[2])
    )


def convert_img_dim(image):
    return image.reshape(image.shape[1], image.shape[2], image.shape[0])

