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
from scripts.rgb_manager import RGBImageManager
from scripts.utils import get_overlay_rgb_image
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent.resolve()


def get_image(file_name, rgb_manager):
    image_bgr = cv2.imread(file_name)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_undistorted = rgb_manager.correction(image)
    return image_undistorted


def infer_image(image, inference):
    mask_image_tmp = convert_img_dim(inference(add_dummy_dim(image)))
    mask_image = (mask_image_tmp*255).astype(np.uint8)
    return mask_image


def get_image_pathes(input_data_dir):
    image_pathes = Path(input_data_dir).glob("*.png")
    image_path_list = [str(image_path) for image_path in image_pathes]
    return image_path_list



@click.command()
@click.option("--input-data-dir", "-i", default=f"{SCRIPT_DIR}/data")
@click.option("--output-data-dir", "-o", default=f"{SCRIPT_DIR}/output")
def main(input_data_dir, output_data_dir):
    inference = create_inference()
    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir)

    image_path_list = get_image_pathes(input_data_dir)

    for image_path in tqdm(image_path_list):
        base_name = Path(image_path).name
        rgb_image = cv2.imread(image_path)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        segmentation_mask = infer_image(bgr_image, inference)
        rgb_image_masked = get_overlay_rgb_image(bgr_image, segmentation_mask)

        cv2.imwrite(f"{output_data_dir}/{base_name}", rgb_image_masked)
        cv2.waitKey(10)

if __name__ == "__main__":
    main()
