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
from torch.nn import functional as F

import torch
import tensorrt as trt
import pycuda.driver as cuda # pycuda 2018.1.1
import pycuda.autoinit
import onnx
import onnx_tensorrt.backend as backend
import onnxruntime

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
    mask_image = (mask_image_tmp[:,:,0]*255).astype(np.uint8)
    return mask_image


def get_image_pathes(input_data_dir):
    image_pathes = Path(input_data_dir).glob("*.png")
    image_path_list = [str(image_path) for image_path in image_pathes]
    return image_path_list

def transform_image(imgs_raw, depth=5):
    imgs = torch.from_numpy(imgs_raw.transpose(0, 3, 1, 2)).clone()   # BHWC -> BCHW
    imgs = imgs.to('cuda')

    resize=1024
    pad_unit = 2 ** 5

    _PAD_VALUE = 114  # same value as YOLO
    _NORM = torch.tensor([255, 255, 255], dtype=torch.float32).to('cuda')        # for uint8
    _MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).to('cuda')  # mean of imagenet
    _STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).to('cuda')   # std of imagenet

    # resize
    b, ch, h, w = imgs.shape
    resize_ratio = resize / max(h, w)
    imgs_resized = F.interpolate(imgs.float(), scale_factor=(resize_ratio, resize_ratio), mode='bilinear', align_corners=False)

    # pad
    b, ch, rh, rw = imgs_resized.shape
    pad_w = 0 if (rw % pad_unit) == 0 else (pad_unit - (rw % pad_unit)) // 2
    pad_h = 0 if (rh % pad_unit) == 0 else (pad_unit - (rh % pad_unit)) // 2
    # pad_w = (self.resize - rw) // 2
    # pad_h = (self.resize - rh) // 2
    pad_w += pad_unit // 2
    pad_h += pad_unit // 2
    imgs_pdded = F.pad(input=imgs_resized, pad=[pad_w, pad_w, pad_h, pad_h], mode='constant', value=_PAD_VALUE)

    # normalize
    imgs_normalized = ((imgs_pdded.permute(0, 2, 3, 1) / _NORM) - _MEAN) / _STD  # BCHW -> BHWC
    imgs_transformed = imgs_normalized.permute(0, 3, 1, 2).contiguous()                         # BHWC -> BCHW

    return imgs_transformed


@click.command()
@click.option("--input-data-dir", "-i", default=f"{SCRIPT_DIR}/data")
@click.option("--output-data-dir", "-o", default=f"{SCRIPT_DIR}/output")
@click.option("--onnx-name", "-c", default=f"{SCRIPT_DIR}/model.onnx")
def main(input_data_dir, output_data_dir, onnx_name):
    sess = onnxruntime.InferenceSession(onnx_name)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    print("The model expects input shape: ", sess.get_inputs()[0].shape)

    image_path_list = get_image_pathes(input_data_dir)
    for image_path in tqdm(image_path_list):
        base_name = Path(image_path).name
        rgb_image = cv2.imread(image_path)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        #test = transform_image(add_dummy_dim(bgr_image))        
        result = sess.run(None, {input_name: np.random.rand(1, 3, 1024, 1024).astype(np.float32)})

        prob = result[0]
        #import pdb; pdb.set_trace()
        '''
        mask_image_tmp = prob[0].argmax(0)
        segmentation_mask = (mask_image_tmp*255).astype(np.uint8)
        rgb_image_masked = get_overlay_rgb_image(bgr_image, segmentation_mask)
        cv2.imwrite(f"{output_data_dir}/{base_name}", rgb_image_masked)
        cv2.waitKey(10)
        '''

if __name__ == "__main__":
    main()
