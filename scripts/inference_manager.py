import numpy as np
import torch
from torch.nn import functional as F
import pytorch_lightning as pl
import segmentation_models_pytorch as smp


class InferSegmentation:
    def __init__(
        self, weights: str, architecture="Unet", encoder="resnet34", depth=5, in_channels=3, classes=2, activation="softmax", resize=1024, gpu=0
    ):
        self.weights = weights
        self.architecture = architecture
        self.encoder = encoder
        self.depth = depth
        self.in_channels = in_channels
        self.classes = classes
        self.activation = activation
        self.resize = resize
        self.gpu = gpu

        # define a model
        self.model = SegmentationModels(
            architecture=self.architecture,
            encoder=self.encoder,
            depth=self.depth,
            in_channels=self.in_channels,
            classes=self.classes,
            activation=self.activation,
        )

        # set a device
        if self.gpu >= 0:
            self.map_location = "cuda:" + str(self.gpu)
        else:
            self.map_location = "cpu"
        print("running on %s" % self.map_location)

        # load weights
        checkpoint = torch.load(self.weights, map_location=self.map_location)
        self.model.load_state_dict(checkpoint["state_dict"], strict=False)
        # self.model.load_state_dict(checkpoint['state_dict'])
        if self.gpu >= 0:
            self.model.cuda(self.gpu)

        # set inference mode
        self.model.eval()
        self.model.freeze()
        torch.backends.cudnn.deterministic = True

        # set constant values
        self._PAD_VALUE = 114  # same value as YOLO
        self._NORM = torch.tensor([255, 255, 255], dtype=torch.float32).to(self.map_location)  # for uint8
        self._MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).to(self.map_location)  # mean of imagenet
        self._STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).to(self.map_location)  # std of imagenet

    # input:RGB, uint8, [b, h, w, c]
    # output:class indices, uint8, [b, h, w]
    def __call__(self, imgs):
        assert imgs.ndim == 4, "imgs.ndim=={0}, actual {1}".format(4, imgs.ndim)
        assert imgs.dtype == "uint8", "imgs.dtype=={0}, actual {1}".format("uint8", imgs.dtype)

        self.batch, self.height, self.width, self.channel = imgs.shape
        assert self.channel == self.in_channels, "imgs_channel=={0}, actual {1}".format(self.in_channels, self.channel)

        with torch.no_grad():
            # convert to tensor and transfer to gpu
            imgs_tensor = torch.from_numpy(imgs.transpose(0, 3, 1, 2)).clone()  # BHWC -> BCHW
            imgs_tensor = imgs_tensor.to(self.map_location)

            # transform imgs
            imgs_transformed = self._transform_imgs(imgs_tensor)

            # infer imgs
            outputs_tensor = self.model(imgs_transformed)
            _, masks_tensor = torch.max(outputs_tensor, dim=1)  # convert to class indices

            # reverse transform masks
            masks_rev_transformed = self._rev_transform_masks(masks_tensor)

            # convert to numpy and transfer to cpu
            masks = masks_rev_transformed.to("cpu").detach().numpy().copy()

        return masks

    def _transform_imgs(self, imgs):
        # resize
        b, ch, h, w = imgs.shape
        self.resize_ratio = self.resize / max(h, w)
        imgs_resized = F.interpolate(imgs.float(), scale_factor=(self.resize_ratio, self.resize_ratio), mode="bilinear", align_corners=False)

        # pad
        b, ch, rh, rw = imgs_resized.shape
        pad_w = 0 if (rw % self.model.pad_unit) == 0 else (self.model.pad_unit - (rw % self.model.pad_unit)) // 2
        pad_h = 0 if (rh % self.model.pad_unit) == 0 else (self.model.pad_unit - (rh % self.model.pad_unit)) // 2
        # pad_w = (self.resize - rw) // 2
        # pad_h = (self.resize - rh) // 2
        pad_w += self.model.pad_unit // 2
        pad_h += self.model.pad_unit // 2
        imgs_pdded = F.pad(input=imgs_resized, pad=[pad_w, pad_w, pad_h, pad_h], mode="constant", value=self._PAD_VALUE)

        # normalize
        imgs_normalized = ((imgs_pdded.permute(0, 2, 3, 1) / self._NORM) - self._MEAN) / self._STD  # BCHW -> BHWC
        imgs_transformed = imgs_normalized.permute(0, 3, 1, 2).contiguous()  # BHWC -> BCHW

        return imgs_transformed

    def _rev_transform_masks(self, masks):
        # resize masks to the original size
        b, h, w = masks.shape
        resize_ratio = 1 / self.resize_ratio
        masks = torch.reshape(masks, (b, 1, h, w))
        masks_resized = F.interpolate(masks.float(), scale_factor=(resize_ratio, resize_ratio), mode="nearest")
        b, _, rh, rw = masks_resized.shape
        masks_resized = torch.reshape(masks_resized, (b, rh, rw)).type(torch.uint8)

        # remove padding
        tl_y = (rh - self.height) // 2
        tl_x = (rw - self.width) // 2
        masks_cropped = masks_resized[:, tl_y : tl_y + self.height, tl_x : tl_x + self.width]

        return masks_cropped


# LightningModule
class SegmentationModels(pl.LightningModule):
    def __init__(self, architecture="Unet", encoder="resnet34", depth=5, in_channels=3, classes=2, activation="softmax"):
        super(SegmentationModels, self).__init__()
        self.architecture = architecture
        self.encoder = encoder
        self.depth = depth
        self.in_channels = in_channels
        self.classes = classes
        self.activation = activation

        # define model
        _ARCHITECTURES = ["Unet", "Linknet", "FPN", "PSPNet", "PAN", "DeepLabV3", "DeepLabV3Plus"]
        assert self.architecture in _ARCHITECTURES, "architecture=={0}, actual '{1}'".format(_ARCHITECTURES, self.architecture)
        if self.architecture == "Unet":
            self.model = smp.Unet(
                encoder_name=self.encoder,
                encoder_weights=None,
                encoder_depth=self.depth,
                in_channels=self.in_channels,
                classes=self.classes,
                activation=self.activation,
            )
            self.pad_unit = 2 ** self.depth
        elif self.architecture == "Linknet":
            self.model = smp.Linknet(
                encoder_name=self.encoder,
                encoder_weights=None,
                encoder_depth=self.depth,
                in_channels=self.in_channels,
                classes=self.classes,
                activation=self.activation,
            )
            self.pad_unit = 2 ** self.depth
        elif self.architecture == "FPN":
            self.model = smp.FPN(
                encoder_name=self.encoder,
                encoder_weights=None,
                encoder_depth=self.depth,
                in_channels=self.in_channels,
                classes=self.classes,
                activation=self.activation,
            )
            self.pad_unit = 2 ** self.depth
        elif self.architecture == "PSPNet":
            self.model = smp.PSPNet(
                encoder_name=self.encoder,
                encoder_weights=None,
                encoder_depth=self.depth,
                in_channels=self.in_channels,
                classes=self.classes,
                activation=self.activation,
            )
            self.pad_unit = 2 ** self.depth
        elif self.architecture == "PAN":
            self.model = smp.PAN(
                encoder_name=self.encoder,
                encoder_weights=None,
                encoder_depth=self.depth,
                in_channels=self.in_channels,
                classes=self.classes,
                activation=self.activation,
            )
            self.pad_unit = 2 ** self.depth
        elif self.architecture == "DeepLabV3":
            self.model = smp.DeepLabV3(
                encoder_name=self.encoder,
                encoder_weights=None,
                encoder_depth=self.depth,
                in_channels=self.in_channels,
                classes=self.classes,
                activation=self.activation,
            )
            self.pad_unit = 2 ** self.depth
        elif self.architecture == "DeepLabV3Plus":
            self.model = smp.DeepLabV3Plus(
                encoder_name=self.encoder,
                encoder_weights=None,
                encoder_depth=self.depth,
                in_channels=self.in_channels,
                classes=self.classes,
                activation=self.activation,
            )
            self.pad_unit = 2 ** self.depth

    def forward(self, x):
        x = self.model(x)
        return x