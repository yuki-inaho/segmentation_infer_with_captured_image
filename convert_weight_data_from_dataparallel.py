import click
import torch
from pathlib import Path
import numpy as np
import toml
from collections import OrderedDict

SCRIPT_DIR = Path(__file__).parent.resolve()


def fix_model_state_dict_from_model_parallel(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name == "state_dict":
            model_state_dict = state_dict["state_dict"]
            new_state_dict["state_dict"] = {}
            for mk, mv in model_state_dict.items():
                model_name = mk
                if model_name.startswith("model.module."):
                    model_name = model_name.replace("model.module.", "model.")
                    new_state_dict["state_dict"][model_name] = mv
                else:
                    new_state_dict["state_dict"][mk] = mv
        else:
            new_state_dict[k] = v
    return new_state_dict


@click.command()
@click.option("--config-name", "-c", default=f"{SCRIPT_DIR}/cfg/semantic_segmentation_multi_class.toml")
@click.option("--output-model-file", "-o", default=f"converted.pth")
def main(config_name, output_model_file):
    toml_dict = toml.load(open(config_name, "r"))
    weight_file_path = toml_dict["weights"]
    checkpoint = torch.load(weight_file_path)
    checkpoint_fixed = fix_model_state_dict_from_model_parallel(checkpoint)
    torch.save(checkpoint_fixed, output_model_file)


if __name__ == "__main__":
    main()