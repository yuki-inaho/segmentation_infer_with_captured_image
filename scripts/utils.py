import sys
import toml
from pathlib import Path
from scripts.inference_manager import InferSegmentation
import numpy as np
import cv2

BASE_DIR_PATH = Path(__file__).resolve().parents[1]


def create_inference(config_path=f"{BASE_DIR_PATH}/cfg/semantic_segmentation.toml"):
    if len(config_path) == 0:
        raise "no config file given"
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


def get_overlay_rgb_image(rgb_image, mask, rgb_rate=0.6, mask_rate=0.4):
    if len(mask.shape) > 2:
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        segmentation_overlay_rgb = cv2.addWeighted(rgb_image, rgb_rate, mask, mask_rate, 2.5)
    else:
        mask_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        nonzero_idx = np.where(mask>0)
        mask_image[nonzero_idx[0], nonzero_idx[1], :] = (0, 0, 255)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        segmentation_overlay_rgb = cv2.addWeighted(rgb_image, rgb_rate, mask_image, mask_rate, 2.5)
    return segmentation_overlay_rgb