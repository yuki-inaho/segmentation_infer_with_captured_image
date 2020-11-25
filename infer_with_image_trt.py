import os
import cv2
from scripts.utils import create_inference, add_dummy_dim, convert_img_dim
import click
from pathlib import Path
import numpy as np

import torch

from torch2trt import torch2trt
import tensorrt as trt
import pycuda.driver as cuda  # pycuda 2018.1.1
import pycuda.autoinit
import onnx
import onnx_tensorrt.backend as backend

from scripts.rgb_manager import RGBImageManager
from scripts.utils import get_overlay_rgb_image
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent.resolve()

import pycuda.autoinit
import pycuda.driver as cuda


def build_engine(model_file, max_ws=512 * 1024 * 1024, fp16=False):
    print("building engine")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    builder.fp16_mode = fp16
    config = builder.create_builder_config()
    config.max_workspace_size = max_ws
    if fp16:
        config.flags |= 1 << int(trt.BuilderFlag.FP16)

    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    with trt.OnnxParser(network, TRT_LOGGER) as parser:
        with open(model_file, "rb") as model:
            parsed = parser.parse(model.read())
            print("network.num_layers", network.num_layers)
            # last_layer = network.get_layer(network.num_layers - 1)
            # network.mark_output(last_layer.get_output(0))
            engine = builder.build_engine(network, config=config)
            return engine


def get_image(file_name, rgb_manager):
    image_bgr = cv2.imread(file_name)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_undistorted = rgb_manager.correction(image)
    return image_undistorted


def infer_image(image, inference):
    mask_image_tmp = convert_img_dim(inference(add_dummy_dim(image)))
    mask_image = (mask_image_tmp[:, :, 0] * 255).astype(np.uint8)
    return mask_image


def get_image_pathes(input_data_dir):
    image_pathes = Path(input_data_dir).glob("*.png")
    image_path_list = [str(image_path) for image_path in image_pathes]
    return image_path_list


def resize_with_padding(im, desired_size=1024):
    old_size = im.shape[:2]
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return new_im


@click.command()
@click.option("--input-data-dir", "-i", default=f"{SCRIPT_DIR}/data")
@click.option("--output-data-dir", "-o", default=f"{SCRIPT_DIR}/output")
@click.option("--config-name", "-c", default=f"{SCRIPT_DIR}/cfg/semantic_segmentation.toml")
@click.option("--onnx-name", "-m", default=f"{SCRIPT_DIR}/model.onnx")
def main(input_data_dir, output_data_dir,config_name, onnx_name):
    inference = create_inference(config_path=config_name)
    inference.model.model.segmentation_head[2].activation.dim = 1
    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir)

    # create example data
    x = torch.ones((1, 3, 1024, 1024)).cuda().half()
    model_trt = torch2trt(inference.model.half(), [x], fp16_mode=True, max_batch_size=1, max_workspace_size=1<<30, keep_network=False)
    del inference

    import time
    image_path_list = get_image_pathes(input_data_dir)
    for image_path in tqdm(image_path_list):
        base_name = Path(image_path).name
        rgb_image = cv2.imread(image_path)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        x = torch.from_numpy(bgr_image.astype(np.float32)).clone().half()
        start = time.time()
        model_trt(x)
        end = time.time()
        print(end-start)

        #img_tfmd = transform_image(add_dummy_dim(bgr_image))        
        #img_tfmd_ary = img_tfmd.to('cpu').detach().numpy()

        #test = add_dummy_dim(bgr_image)
        #result = sess.run(None, {input_name: np.random.rand(1, 3, 1024, 1024).astype(np.float32)})

    '''
    engine = build_engine(onnx_name)
    context = engine.create_execution_context()
    h_input = cuda.pagelocked_empty(
        trt.volume(engine.get_binding_shape(0)),
        dtype=trt.nptype(engine.get_binding_dtype(0)),
    )
    h_output = cuda.pagelocked_empty(
        trt.volume(engine.get_binding_shape(1)),
        dtype=trt.nptype(engine.get_binding_dtype(1)),
    )
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(trt.volume(engine.get_binding_shape(1)) * 4)
    stream = cuda.Stream()

    image_path_list = get_image_pathes(input_data_dir)
    for image_path in tqdm(image_path_list):
        base_name = Path(image_path).name
        rgb_image = cv2.imread(image_path)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        bgr_image = resize_with_padding(bgr_image)
        input_numpy_array = add_dummy_dim(bgr_image)
        input_numpy_array_shape = input_numpy_array.shape

        np.copyto(h_input, input_numpy_array.flatten())
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async(
            bindings=[int(d_input), int(d_output)], stream_handle=stream.handle
        )
        stream.synchronize()
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        tensor = torch.zeros(input_numpy_array_shape, dtype=torch.float32).cuda()
        cuda.memcpy_dtod(
            tensor.data_ptr(), d_output, trt.volume(engine.get_binding_shape(1)) * 4
        )

        output_numpy_array_flatten = cuda.from_device(
            d_output, 1024 * 1024 * 2, dtype=np.float32
        )
        output_numpy_array = output_numpy_array_flatten.reshape((1024, 1024, 2))

        segmentation_mask = (output_numpy_array[:, :, 1] * 255).astype(np.uint8)

        rgb_image_masked = get_overlay_rgb_image(bgr_image, segmentation_mask)
        cv2.imwrite(f"{output_data_dir}/{base_name}", rgb_image_masked)
        cv2.waitKey(10)
    '''

if __name__ == "__main__":
    main()
