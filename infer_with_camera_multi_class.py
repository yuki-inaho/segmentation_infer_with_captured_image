import cv2
from scripts.utils import create_inference
import click
from pathlib import Path
from scripts.rgb_manager import RGBCaptureManager
from scripts.class_info_manager import ClassInformationManager
from scripts.utils import get_overlay_rgb_image, infer_image


SCRIPT_DIR = Path(__file__).parent.resolve()


def convert_image_to_infer(image_bgr, rgb_manager):
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image


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
            segmentation_mask = infer_image(rgb_image, inference, class_manager)
            rgb_image_masked = get_overlay_rgb_image(rgb_image, segmentation_mask, rgb_rate=rgb_rate, mask_rate=1 - rgb_rate)
            masked_image_resized = cv2.resize(rgb_image_masked, (1280, 720))
            cv2.imshow("Segmentation", masked_image_resized)
            cv2.waitKey(10)
        key = cv2.waitKey(10)
        if key & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()