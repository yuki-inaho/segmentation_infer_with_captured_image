import toml
import numpy as np
from collections import OrderedDict


def get_camera_parameter(toml_path):
    toml_dict = toml.load(open(toml_path))
    toml_decoder = toml.TomlDecoder(_dict=OrderedDict)

    intrinsic = IntrinsicParameter()
    intrinsic.set_intrinsic_parameter(*[toml_dict["Rgb"][elem] for elem in ["fx", "fy", "cx", "cy"]])
    dist_coeffs = [toml_dict["Rgb"][elem] for elem in ["k1", "k2", "k3", "k4"]]
    return intrinsic, dist_coeffs

class IntrinsicParameter:
    def __init__(self):
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

    def set_intrinsic_parameter(self, fx, fy, cx, cy):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def set_image_size(self, image_width, image_height):
        self._image_width = image_width
        self._image_height = image_height

    @property
    def K(self):
        return np.array([
            [self.fx, 0., self.cx],
            [0., self.fy, self.cy],
            [0., 0., 1.]
        ])

    @property
    def center(self):
        return self.cx, self.cy

    @property
    def focal(self):
        return self.fx, self.fy

    @property
    def width(self):
        return self._image_width

    @property
    def height(self):
        return self._image_height