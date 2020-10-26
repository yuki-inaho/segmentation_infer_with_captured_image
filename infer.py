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


@click.command()
@click.option("--toml-path", "-t", default=f"{SCRIPT_DIR}/cfg/dualzense_r.toml")
@click.option("--input-data-dir", "-i", default=f"{SCRIPT_DIR}/data")
@click.option("--output-data-dir", "-o", default=f"{SCRIPT_DIR}/output")
def main(toml_path, input_data_dir, output_data_dir):
    rgb_manager = RGBImageManager(toml_path)
    # = get_dataset(input_data_dir)
    inference = create_inference()

    file_name = f"{SCRIPT_DIR}/data/2020-10-09_05-31-30_DUALZENSE_R_r_see_0.png"
    base_name = Path(file_name).name
    rgb_image = get_image(file_name, rgb_manager)
    segmentation_mask = infer_image(rgb_image, inference)
    rgb_image_masked = get_overlay_rgb_image(rgb_image, segmentation_mask)

    cv2.imwrite(f"{output_data_dir}/{base_name}", rgb_image_masked)
    cv2.waitKey(10)

if __name__ == "__main__":
    main()
