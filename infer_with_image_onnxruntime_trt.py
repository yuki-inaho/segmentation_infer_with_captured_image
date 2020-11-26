import os
import cv2
from scripts.utils import (
    create_inference,
    add_dummy_dim,
    convert_img_dim
)
import click
import time
from pathlib import Path
import numpy as np
from torch.nn import functional as F
import onnx_tensorrt.backend as backend

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
    pad_w = (resize - rw) // 2
    pad_h = (resize - rh) // 2
    #pad_w += pad_unit // 2
    #pad_h += pad_unit // 2
    imgs_pdded = F.pad(input=imgs_resized, pad=[pad_w, pad_w, pad_h, pad_h], mode='constant', value=_PAD_VALUE)

    # normalize
    imgs_normalized = ((imgs_pdded.permute(0, 2, 3, 1) / _NORM) - _MEAN) / _STD  # BCHW -> BHWC
    imgs_transformed = imgs_normalized.permute(0, 3, 1, 2).contiguous()                         # BHWC -> BCHW

    return imgs_transformed



def build_engine(model_file, max_ws=384 * 1024 * 1024, fp16=False):
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

def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    [cuda.memcpy_htod(inp.device, inp.host) for inp in inputs]
    context.execute(batch_size=batch_size, bindings=bindings)
    [cuda.memcpy_dtoh(out.host, out.device) for out in outputs]
    return [out.host for out in outputs]

        
@click.command()
@click.option("--input-data-dir", "-i", default=f"{SCRIPT_DIR}/data")
@click.option("--output-data-dir", "-o", default=f"{SCRIPT_DIR}/output")
@click.option("--onnx-name", "-c", default=f"{SCRIPT_DIR}/model.onnx")
def main(input_data_dir, output_data_dir, onnx_name):
    #sess = onnxruntime.InferenceSession(onnx_name)

    '''
    so = onnxruntime.SessionOptions()
    so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENoABLE_ALL
    sess = onnxruntime.InferenceSession(onnx_name, sess_options=so)
    sess.set_providers(['CUDAExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    '''

    saved=False
    if not saved:
        model = onnx.load(onnx_name)
        #engine = build_engine(onnx_name)
        engine = backend.prepare(model, device="CUDA:0", max_batch_size=1,
                                 max_workspace_size=1<<30)
        with open('./engine.trt', 'wb') as f:
            f.write(bytearray(engine.engine.engine.serialize()))
        result = engine.run(np.random.rand(1, 3, 1024, 1024).astype(np.float16))[0]
    else:
        TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
        runtime = trt.Runtime(TRT_LOGGER)
        with open('./engine.trt', 'rb') as f:
            engine_bytes = f.read()
            engine = runtime.deserialize_cuda_engine(engine_bytes)
        context = engine.create_execution_context()
        h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(engine.get_binding_dtype(0)))
        h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(engine.get_binding_dtype(1)))
        d_input = cuda.mem_alloc(h_input.nbytes)
        d_output = cuda.mem_alloc(trt.volume(engine.get_binding_shape(1)) * 4)
        stream = cuda.Stream()
        
    
    image_path_list = get_image_pathes(input_data_dir)
    for image_path in tqdm(image_path_list):
        base_name = Path(image_path).name
        rgb_image = cv2.imread(image_path)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        img_tfmd = transform_image(add_dummy_dim(bgr_image))        
        img_tfmd_ary = img_tfmd.to('cpu').detach().numpy()

        #test = add_dummy_dim(bgr_image)
        #result = sess.run(None, {input_name: np.random.rand(1, 3, 1024, 1024).astype(np.float32)})

        start = time.time()
        if saved:
            np.copyto(h_input, img_tfmd_ary.astype(np.float16))
            cuda.memcpy_htod_async(d_input, h_input, stream)
            context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
            stream.synchronize()
            cuda.memcpy_dtoh_async(h_output, d_output, stream)
        else:
            result = engine.run(img_tfmd_ary.astype(np.float16))[0]
        #result = sess.run(None, {input_name: img_tfmd_ary.astype(np.float16)})
        end = time.time()
        print(end-start)

        prob = result[0][0]
        label_img = (prob.argmax(0)*255).astype(np.uint8)
        
        cv2.imwrite(str(Path(output_data_dir, base_name)), label_img)
        cv2.waitKey(10)

if __name__ == "__main__":

    main()
