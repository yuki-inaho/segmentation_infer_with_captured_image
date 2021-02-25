import cv2
from pathlib import Path
import os
import click

SCRIPT_DIR = str(Path(__file__).parent)

@click.command()
@click.option("--input-data-dir", "-i", default=f"{SCRIPT_DIR}/data")
@click.option("--mask-data-dir", "-m", default=f"{SCRIPT_DIR}/output")
@click.option("--fused-data-dir", "-f", default=f"{SCRIPT_DIR}/fused")
def main(input_data_dir, mask_data_dir, fused_data_dir):
    if not os.path.exists(fused_data_dir):
        os.makedirs(fused_data_dir)

    raw_image_list = [str(path) for path in Path(input_data_dir).glob("*.png")]
    for raw_image_path in raw_image_list:
        image_name = Path(raw_image_path).name
        raw_image_resized = cv2.resize(cv2.imread(raw_image_path), (1280, 720))
        mask_image_path = str(Path(mask_data_dir, image_name))
        mask_image_resized = cv2.resize(cv2.imread(mask_image_path), (1280, 720))
        image_fused = cv2.vconcat([raw_image_resized, mask_image_resized])
        cv2.imwrite(str(Path(fused_data_dir, image_name)), image_fused)
        cv2.waitKey(10)

if __name__ == "__main__":
    main()
