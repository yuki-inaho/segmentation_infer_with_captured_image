import cv2
import toml
import numpy as np
from scripts.camera_parameter import IntrinsicParameter
from pathlib import Path
import pdb

class LensUndistorter:
    def __init__(self, K_rgb, distortion_params, image_width, image_height, enable_tps=False):
        self.distortion_params = distortion_params
        self.DIM = (image_width, image_height)
        if enable_tps:
            self.K_rgb = K_rgb
            _map1, _map2 = cv2.fisheye.initUndistortRectifyMap(
                self.K_rgb, self.distortion_params, np.eye(3),
                self.K_rgb, self.DIM, cv2.CV_16SC2
            )
        else:
            self.K_rgb_raw = K_rgb
            self.K_rgb = cv2.getOptimalNewCameraMatrix(
                self.K_rgb_raw, self.distortion_params,
                self.DIM, 0
            )[0]
            _map1, _map2 = cv2.fisheye.initUndistortRectifyMap(
                self.K_rgb_raw, self.distortion_params, np.eye(3),
                self.K_rgb, self.DIM, cv2.CV_16SC2
            )
        self.map1 = _map1
        self.map2 = _map2
        self.P_rgb = (self.K_rgb[0][0], 0., self.K_rgb[0][2], 0.,
                      0., self.K_rgb[1][1], self.K_rgb[1][2], 0.,
                      0., 0., 1., 0.)

    def correction(self, image):
        return cv2.remap(image, self.map1, self.map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def correction_with_mask(self, mask):
        return cv2.remap(mask, self.map1, self.map2, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

    @property
    def K(self):
        return self.K_rgb

    @property
    def P(self):
        return self.P_rgb


class RGBCaptureManager:
    def __init__(self, toml_path, enable_undistortion=True):
        self._setting(toml_path)
        '''
        Read toml setting file and set
            self.device_id, self.fps
            self.image_{width, height}
            self.image_{width_raw, height_raw}
            self.intrinsic_params, self.intrinsic_params_raw
            self.K_rgb, self.K_rgb_raw,
            self.tps_is_ready = False
        '''
        self.stopped = False
        self.is_grabbed = False
        self.frame = None

        # Set Video Capture Module
        self.cap = cv2.VideoCapture(self.device_id)
        if self.enable_tps:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_width_raw)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_height_raw)
        else:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_height)
        self.cap.set(cv2.CAP_PROP_FPS, int(self.fps))

        self.enable_undistortion = enable_undistortion
        if self.enable_undistortion:
            if self.enable_tps:
                self.lens_undistorter = LensUndistorter(
                    self.K_rgb_raw, self.distortion_params,
                    self.image_width_raw, self.image_height_raw, enable_tps=True
                )
            else:
                # Enable barrel distortion correction, and Disable TPS undistortion case
                self.lens_undistorter = LensUndistorter(
                    self.K_rgb, self.distortion_params,
                    self.image_width, self.image_height, enable_tps=False
                )
        self.P_rgb = np.c_[self.K_rgb, np.repeat(1.0, 3)]

    def _setting(self, toml_path):
        toml_dict = toml.load(open(toml_path))
        # TODO : TPS flag should be more explicitly
        undistortion_table_existence = "Rgb_Undistortion" in toml_dict.keys()
        if not undistortion_table_existence:
            self.enable_tps = False
        else:
            self.enable_tps = self.ready_for_tps(toml_dict)

        # Set Camera Settings
        self.device_id = toml_dict["Rgb"]["device_id"]
        self.image_width = toml_dict["Rgb"]["width"]
        self.image_height = toml_dict["Rgb"]["height"]
        if self.enable_tps:
            self.image_width_raw = toml_dict["Rgb_Undistortion"]["width_raw"]
            self.image_height_raw = toml_dict["Rgb_Undistortion"]["height_raw"]
        self.fps = toml_dict["Rgb"]["fps"]

        # Set Camera Parameters
        intrinsic_elems = ["fx", "fy", "cx", "cy"]
        self.intrinsic_params = IntrinsicParameter()
        self.intrinsic_params.set_intrinsic_parameter(
            *[toml_dict["Rgb"][elem] for elem in intrinsic_elems]
        )
        self.intrinsic_params.set_image_size(
            *[toml_dict["Rgb"][elem] for elem in ["width", "height"]])
        self.K_rgb = np.array(
            [[self.intrinsic_params.fx, 0, self.intrinsic_params.cx],
             [0, self.intrinsic_params.fy, self.intrinsic_params.cy],
             [0, 0, 1]]
        )
        if self.enable_tps:
            self.modify_camera_parameter()
            self.tps_is_ready = True
        else:
            self.tps_is_ready = False

        self.distortion_params = np.array(
            [toml_dict["Rgb"]["k{}".format(i+1)] for i in range(4)]
        )

    def ready_for_tps(self, toml_dict):
        _enable_tps = toml_dict["Rgb_Undistortion"]["enable_tps"]
        if _enable_tps:
            tps_file_path = Path(toml_dict["Rgb_Undistortion"]["tps_file_path"])
            if tps_file_path.exists():
                distortion_info_mat = np.load(str(tps_file_path))
                self.enable_tps_undistortion(distortion_info_mat)
            else:
                print("TPS Parameter File doesn't exist. TPS undistortion disabled")
                _enable_tps = False
        else:
            _enable_tps = False
        return _enable_tps

    def modify_camera_parameter(self):
        # When TPS undistortion enabled, it is necessary to combine 1080p barrel undistortion
        # In the see3cam case, focal lengthes and distortion coefficients is the same between 720p and 1080p FoV
        # On below procedure,  principal points on 1080p FoV is generated using 720p ones.
        diff_x = float(self.image_width)/2 - self.intrinsic_params.cx
        diff_y = float(self.image_height)/2 - self.intrinsic_params.cy
        self.cx_raw = float(self.image_width_raw)/2.0 + diff_x
        self.cy_raw = float(self.image_height_raw)/2.0 + diff_x
        self.intrinsic_params.cx = float(self.image_width)/2
        self.intrinsic_params.cy = float(self.image_height)/2
        self.intrinsic_raw = IntrinsicParameter()
        self.intrinsic_raw.set_intrinsic_parameter(
            self.intrinsic_params.fx, self.intrinsic_params.fy, self.cx_raw, self.cy_raw
        )
        self.K_rgb = np.array(
            [[self.intrinsic_params.fx, 0, self.intrinsic_params.cx],
             [0, self.intrinsic_params.fy, self.intrinsic_params.cy],
             [0, 0, 1]]
        )
        self.K_rgb_raw = np.array(
            [[self.intrinsic_raw.fx, 0, self.intrinsic_raw.cx],
             [0, self.intrinsic_raw.fy, self.intrinsic_raw.cy],
             [0, 0, 1]]
        )

    def update(self):
        # For latency related with buffer
        for i in range(5):
            status_rgb, rgb_image = self.cap.read()
        if not status_rgb:
            return False
        if self.enable_undistortion:
            if self.tps_is_ready:
                rgb_img_barrel_undist = self.lens_undistorter.correction(rgb_image)
                rgb_img_undist = self.distortion_correction_with_tps(rgb_img_barrel_undist)
                rgb_img_undist = self.clop(rgb_img_undist)
                self.frame = rgb_img_undist.astype(np.uint8)
            else:
                self.frame = self.lens_undistorter.correction(rgb_image)
        else:
            self.frame = rgb_image
        return True

    def read(self):
        return self.frame

    def enable_tps_undistortion(self, distortion_info_mat):
        self.distortion_info_mat = distortion_info_mat
        self.tps_is_ready = True

    def distortion_correction_with_tps(self, rgb_image):
        # TODO: remove redundancy of clipping process
        idxs = np.arange(self.image_width_raw*self.image_height_raw)
        _xidxs = np.tile(np.arange(self.image_width_raw),
                         self.image_height_raw)
        _yidxs = np.repeat(np.arange(self.image_height_raw),
                           self.image_width_raw)
        xidxs = _xidxs - self.distortion_info_mat[:, 0].astype(np.int32)
        xidxs[xidxs < 0] = 0
        xidxs[xidxs >= self.image_width_raw] = self.image_width_raw - 1
        yidxs = _yidxs - self.distortion_info_mat[:, 1].astype(np.int32)
        yidxs[yidxs < 0] = 0
        yidxs[yidxs >= self.image_height_raw] = self.image_height_raw - 1
        distination_idxs = np.c_[xidxs, yidxs]
        rgb_image_undistorted = rgb_image[yidxs, xidxs, :].reshape(
            (self.image_height_raw, self.image_width_raw, 3))
        return rgb_image_undistorted

    def clop(self, rgb_image):
        x_clopped = (
            self.intrinsic_raw.cx + np.arange(self.image_width) - float(self.image_width)/2
        ).astype(np.int32)
        y_clopped = (
            self.intrinsic_raw.cy + np.arange(self.image_height) - float(self.image_height)/2
        ).astype(np.int32)
        return rgb_image[y_clopped, :, :][:, x_clopped, :]

    def get_param_matrix(self):
        return self.K_rgb, self.P_rgb

    def get_intrinsic_parameters(self):
        return self.intrinsic_params

    @property
    def K(self):
        return self.K_rgb

    @property
    def K_raw(self):
        return self.K_rgb_raw


class RGBImageManager:
    def __init__(self, toml_path, enable_undistortion=True):
        self._setting(toml_path)
        '''
        Read toml setting file and set
            self.device_id, self.fps
            self.image_{width, height}
            self.image_{width_raw, height_raw}
            self.intrinsic_params, self.intrinsic_params_raw
            self.K_rgb, self.K_rgb_raw,
            self.tps_is_ready = False
        '''
        self.stopped = False
        self.is_grabbed = False
        self.frame = None

        self.enable_undistortion = enable_undistortion
        if self.enable_undistortion:
            # Enable barrel distortion correction, and Disable TPS undistortion case
            self.lens_undistorter = LensUndistorter(
                self.K_rgb, self.distortion_params,
                self.image_width, self.image_height, enable_tps=False
            )
        self.P_rgb = np.c_[self.K_rgb, np.repeat(1.0, 3)]

    def _setting(self, toml_path):
        toml_dict = toml.load(open(toml_path))

        # Set Camera Settings
        self.device_id = toml_dict["Rgb"]["device_id"]
        self.image_width = toml_dict["Rgb"]["width"]
        self.image_height = toml_dict["Rgb"]["height"]
        self.fps = toml_dict["Rgb"]["fps"]

        # Set Camera Parameters
        intrinsic_elems = ["fx", "fy", "cx", "cy"]
        self.intrinsic_params = IntrinsicParameter()
        self.intrinsic_params.set_intrinsic_parameter(
            *[toml_dict["Rgb"][elem] for elem in intrinsic_elems]
        )
        self.intrinsic_params.set_image_size(
            *[toml_dict["Rgb"][elem] for elem in ["width", "height"]])
        self.K_rgb = np.array(
            [[self.intrinsic_params.fx, 0, self.intrinsic_params.cx],
             [0, self.intrinsic_params.fy, self.intrinsic_params.cy],
             [0, 0, 1]]
        )

        self.distortion_params = np.array(
            [toml_dict["Rgb"]["k{}".format(i+1)] for i in range(4)]
        )

    def correction(self, rgb_img_raw):
        # For latency related with buffer
        rgb_img = rgb_img_raw.copy()
        if self.enable_undistortion:
            return self.lens_undistorter.correction(rgb_img)
        else:
            return rgb_img

    def correction_with_mask(self, mask):
        # For latency related with buffer
        mask = mask.copy()
        if self.enable_undistortion:
            return self.lens_undistorter.correction_with_mask(mask)
        else:
            return mask

    def get_param_matrix(self):
        return self.K_rgb, self.P_rgb

    def get_intrinsic_parameters(self):
        return self.intrinsic_params

    @property
    def K(self):
        return self.K_rgb

    @property
    def K_raw(self):
        return self.K_rgb_raw