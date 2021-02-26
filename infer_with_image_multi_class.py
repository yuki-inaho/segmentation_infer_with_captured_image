import os
import cv2
import click
from pathlib import Path
from scripts.utils import create_inference, get_overlay_rgb_image, infer_image, infer_and_generate_mask_image, get_image_pathes, mkdir_from_path
from scripts.class_info_manager import ClassInformationManager
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent.resolve()


@click.command()
@click.option("--input-data-dir", "-i", default=f"{SCRIPT_DIR}/data")
@click.option("--output-data-dir", "-o", default=f"{SCRIPT_DIR}/output")
@click.option("--generate-only-mask", "-m", is_flag=True)
@click.option("--config-name", "-c", default=f"{SCRIPT_DIR}/cfg/semantic_segmentation_multi_class.toml")
@click.option("--class-definition-json", "-j", default=f"{SCRIPT_DIR}/cfg/classes.json")
def main(input_data_dir, output_data_dir, generate_only_mask, config_name, class_definition_json):
    inference = create_inference(config_path=config_name)
    class_manager = ClassInformationManager(class_definition_json)
    mkdir_from_path(output_data_dir)

    image_path_list = get_image_pathes(input_data_dir)
    for image_path in tqdm(image_path_list):
        base_name = Path(image_path).name
        rgb_image = cv2.imread(image_path)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        if generate_only_mask:
            segmentation_mask = infer_and_generate_mask_image(bgr_image, inference)
            cv2.imwrite(f"{output_data_dir}/{base_name}", segmentation_mask)
        else:
            segmentation_mask = infer_image(bgr_image, inference, class_manager)
            rgb_image_masked = get_overlay_rgb_image(bgr_image, segmentation_mask)
            cv2.imwrite(f"{output_data_dir}/{base_name}", rgb_image_masked)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
