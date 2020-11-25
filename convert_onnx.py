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
import torch
import pdb

SCRIPT_DIR = Path(__file__).parent.resolve()


@click.command()
@click.option("--config-name", "-c", default=f"{SCRIPT_DIR}/cfg/semantic_segmentation.toml")
def main(config_name):
    inference = create_inference(config_path=config_name)
    dummy_input = torch.randn(1, 3, 1024, 1024, device='cpu')
    inference.model.to('cpu')
    inference.model.model.segmentation_head[2].activation.dim = 1
    torch.onnx.export(inference.model, dummy_input, "model.onnx", input_names=["input"], output_names=["output"], verbose=False, opset_version=11, operator_export_type=torch.onnx.OperatorExportTypes.ONNX)


if __name__ == "__main__":
    main()
