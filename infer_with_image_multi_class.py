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


dict_idx2color = {
    0: (0, 0, 0),
    1: (110, 110, 255),  # Oyagi
    2: (150, 249, 152),  # Aspara
    3: (255, 217, 81),  # Ground
    4: (252, 51, 255),  # Tube
    5: (84, 223, 255),  # Pole
}


def get_image(file_name, rgb_manager):
    image_bgr = cv2.imread(file_name)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_undistorted = rgb_manager.correction(image)
    return image_undistorted


def colorize_mask(mask_image, n_label):
    mask_colorized = np.zeros(
        [mask_image.shape[0], mask_image.shape[1], 3], dtype=np.uint8
    )
    for l in range(n_label + 1):
        mask_indices_lth_label = np.where(mask_image == l)
        mask_colorized[mask_indices_lth_label[0], mask_indices_lth_label[1], :] = dict_idx2color[l]
    return mask_colorized


def infer_image(image, inference):
    mask_image_tmp = convert_img_dim(inference(add_dummy_dim(image)))
    #mask_image = (mask_image_tmp*255).astype(np.uint8)
    mask_image = colorize_mask(mask_image_tmp, 5)
    return mask_image


def infer_and_generate_mask_image(image, inference):
    mask_image_tmp = convert_img_dim(inference(add_dummy_dim(image)))
    mask_image = (mask_image_tmp[:,:,0]).astype(np.uint8)
    return mask_image


def get_image_pathes(input_data_dir):
    exts = ['.jpg', '.png']
    image_pathes = sorted([path for path in Path(input_data_dir).rglob('*') if path.suffix.lower() in exts])
    image_path_list = [str(image_path) for image_path in image_pathes]
    return image_path_list


@click.command()
@click.option("--input-data-dir", "-i", default=f"{SCRIPT_DIR}/data")
@click.option("--output-data-dir", "-o", default=f"{SCRIPT_DIR}/output")
@click.option("--generate-only-mask", "-m", is_flag=True)
@click.option("--config-name", "-c", default=f"{SCRIPT_DIR}/cfg/semantic_segmentation_multi_class.toml")
def main(input_data_dir, output_data_dir, generate_only_mask, config_name):
    inference = create_inference(config_path=config_name)
    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir)

    image_path_list = get_image_pathes(input_data_dir)
    for image_path in tqdm(image_path_list):
        base_name = Path(image_path).name
        rgb_image = cv2.imread(image_path)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        if generate_only_mask:
            segmentation_mask = infer_and_generate_mask_image(bgr_image, inference)
            cv2.imwrite(f"{output_data_dir}/{base_name}", segmentation_mask)
        else:
            segmentation_mask = infer_image(bgr_image, inference)
            rgb_image_masked = get_overlay_rgb_image(bgr_image, segmentation_mask)
            cv2.imwrite(f"{output_data_dir}/{base_name}", rgb_image_masked)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
