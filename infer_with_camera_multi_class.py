import os
import cv2
from scripts.utils import create_inference, add_dummy_dim, convert_img_dim
import click
from pathlib import Path
import numpy as np
from scripts.rgb_manager import RGBCaptureManager
from scripts.class_info_manager import ClassInformationManager
from scripts.utils import get_overlay_rgb_image, infer_image


SCRIPT_DIR = Path(__file__).parent.resolve()


<<<<<<< HEAD
dict_idx2color = {
    0: (0, 0, 0),
    1: (110, 110, 255),  # Oyagi
    2: (150, 249, 152),  # Aspara
    3: (255, 217, 81),  # Ground
    4: (252, 51, 255),  # Tube
    5: (84, 223, 255),  # Pole
}


def colorize_mask(mask_image, n_label):
    mask_colorized = np.zeros([mask_image.shape[0], mask_image.shape[1], 3], dtype=np.uint8)
    for l in range(n_label + 1):
        mask_indices_lth_label = np.where(mask_image == l)
        mask_colorized[mask_indices_lth_label[0], mask_indices_lth_label[1], :] = dict_idx2color[l]
    return mask_colorized


=======
>>>>>>> c4c6a3f... Reduce hardcoding lines related with class definition
def convert_image_to_infer(image_bgr, rgb_manager):
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image


<<<<<<< HEAD
def infer_image(image, inference):
    mask_image_tmp = convert_img_dim(inference(add_dummy_dim(image)))
    mask_image = colorize_mask(mask_image_tmp, 5)
    return mask_image


=======
>>>>>>> c4c6a3f... Reduce hardcoding lines related with class definition
@click.command()
@click.option("--toml-path", "-t", default=f"{SCRIPT_DIR}/cfg/camera_parameter.toml")
@click.option("--config-name", "-c", default=f"{SCRIPT_DIR}/cfg/semantic_segmentation_multi_class.toml")
@click.option("--rgb_rate", "-r", default=0.6)
@click.option("--class-definition-json", "-j", default=f"{SCRIPT_DIR}/cfg/classes.json")
def main(toml_path, config_name, rgb_rate, class_definition_json):
    rgb_manager = RGBCaptureManager(toml_path)
    inference = create_inference(config_name)
    class_manager = ClassInformationManager(class_definition_json)

    while True:
        if rgb_manager.update():
            rgb_image = convert_image_to_infer(rgb_manager.read(), rgb_manager)
<<<<<<< HEAD
            segmentation_mask = infer_image(rgb_image, inference)
            rgb_image_masked = get_overlay_rgb_image(rgb_image, segmentation_mask, rgb_rate=rgb_rate, mask_rate=1 - rgb_rate)
=======
            segmentation_mask = infer_image(rgb_image, inference, class_manager)
            rgb_image_masked = get_overlay_rgb_image(
                rgb_image, segmentation_mask,
                rgb_rate=rgb_rate, mask_rate=1-rgb_rate
            )
>>>>>>> c4c6a3f... Reduce hardcoding lines related with class definition
            masked_image_resized = cv2.resize(rgb_image_masked, (1280, 720))
            cv2.imshow("Segmentation", masked_image_resized)
            cv2.waitKey(10)
        key = cv2.waitKey(10)
        if key & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()