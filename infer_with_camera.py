import os
import cv2
from scripts.utils import (
    create_inference,
    add_dummy_dim,
    convert_img_dim
)
import click
from pathlib import Path
import numpy as np
from scripts.rgb_manager import RGBCaptureManager
from scripts.utils import get_overlay_rgb_image


SCRIPT_DIR = Path(__file__).parent.resolve()


def convert_image_to_infer(image_bgr, rgb_manager):
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image


def infer_image(image, inference):
    mask_image_tmp = convert_img_dim(inference(add_dummy_dim(image)))
    mask_image = (mask_image_tmp*255).astype(np.uint8)
    return mask_image


@click.command()
@click.option("--toml-path", "-t", default=f"{SCRIPT_DIR}/cfg/camera.toml")
def main(toml_path):
    rgb_manager = RGBCaptureManager(toml_path)
    inference = create_inference()

    while True:
        if rgb_manager.update():
            rgb_image = convert_image_to_infer(rgb_manager.read(), rgb_manager)
            segmentation_mask = infer_image(rgb_image, inference)
            rgb_image_masked = get_overlay_rgb_image(
                rgb_image, segmentation_mask,
                rgb_rate=0.8, mask_rate=0.2
            )
            masked_image_resized = cv2.resize(rgb_image_masked, (1280, 720))
            cv2.imshow("Segmentation", masked_image_resized)
            cv2.waitKey(10)
        key = cv2.waitKey(10)
        if key & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()