import os
import cv2
import click
from pathlib import Path
import numpy as np
from scripts.class_info_manager import ClassInformationManager
from scripts.utils import colorize_mask
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent.resolve()


def get_image(file_name, rgb_manager):
    image_bgr = cv2.imread(file_name)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_undistorted = rgb_manager.correction(image)
    return image_undistorted


def get_image_pathes(input_data_dir):
    image_pathes = sorted([path for path in Path(input_data_dir).glob("*.png")])
    image_path_list = [str(image_path) for image_path in image_pathes]
    return image_path_list


@click.command()
@click.option("--input-data-dir", "-i", default=f"{SCRIPT_DIR}/output")
@click.option("--output-data-dir", "-o", default=f"{SCRIPT_DIR}/mask_colorized")
@click.option("--class-definition-json", "-c", default=f"{SCRIPT_DIR}/cfg/classes.json")
def main(input_data_dir, output_data_dir, class_definition_json):
    class_manager = ClassInformationManager(class_definition_json)
    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir)

    label_image_path_list = get_image_pathes(input_data_dir)
    for label_image_path in tqdm(label_image_path_list):
        base_name = Path(label_image_path).name
        label_image = cv2.imread(label_image_path, cv2.IMREAD_ANYDEPTH)
        mask_colorized = colorize_mask(label_image, class_manager)
        mask_colorized_path = str(Path(output_data_dir, base_name))
        cv2.imwrite(mask_colorized_path, mask_colorized)
        cv2.waitKey(10)

if __name__ == "__main__":
    main()
