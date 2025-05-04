"""
This module contains classes and functions that are common across both, one-stage
and two-stage detector implementations. You have to implement some parts here -
walk through the notebooks and you will find instructions on *when* to implement
*what* in this module.
"""

from turtle import update
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models import feature_extraction


def hello_common():
    print("Hello from common.py!")


class DetectorBackboneWithFPN(nn.Module):
    r"""
    Detection backbone network: A tiny RegNet model coupled with a Feature
    Pyramid Network (FPN). This model takes in batches of input images with
    shape `(B, 3, H, W)` and gives features from three different FPN levels
    with shapes and total strides upto that level:

        - level p3: (out_channels, H /  8, W /  8)      stride =  8
        - level p4: (out_channels, H / 16, W / 16)      stride = 16
        - level p5: (out_channels, H / 32, W / 32)      stride = 32

    NOTE: We could use any convolutional network architecture that progressively
    downsamples the input image and couple it with FPN. We use a small enough
    backbone that can work with Colab GPU and get decent enough performance.
    """

    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels

        # Initialize with ImageNet pre-trained weights.
        _cnn = models.regnet_x_400mf(weights=models.RegNet_X_400MF_Weights.DEFAULT)

        # Torchvision models only return features from the last level. Detector
        # backbones (with FPN) require intermediate features of different scales.
        # So we wrap the ConvNet with torchvision's feature extractor. Here we
        # will get output features with names (c3, c4, c5) with same stride as
        # (p3, p4, p5) described above.
        self.backbone = feature_extraction.create_feature_extractor(
            _cnn,
            return_nodes={
                "trunk_output.block2": "c3",
                "trunk_output.block3": "c4",
                "trunk_output.block4": "c5",
            },
        )

        # Pass a dummy batch of input images to infer shapes of (c3, c4, c5).
        # Features are a dictionary with keys as defined above. Values are
        # batches of tensors in NCHW format, that give intermediate features
        # from the backbone network.
        dummy_out = self.backbone(torch.randn(2, 3, 224, 224))
        dummy_out_shapes = [(key, value.shape) for key, value in dummy_out.items()]

        print("For dummy input images with shape: (2, 3, 224, 224)")

        for level_name, feature_shape in dummy_out_shapes:
            print(f"Shape of {level_name} features: {feature_shape}")

        self.fpn_params = nn.ModuleDict()

        self.fpn_params["lateral_c3"] = nn.Conv2d(
            dummy_out_shapes[0][1][1],
            self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.fpn_params["lateral_c4"] = nn.Conv2d(
            dummy_out_shapes[1][1][1],
            self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.fpn_params["lateral_c5"] = nn.Conv2d(
            dummy_out_shapes[2][1][1],
            self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.fpn_params["output_c3"] = nn.Conv2d(
            self.out_channels,
            self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.fpn_params["output_c4"] = nn.Conv2d(
            self.out_channels,
            self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.fpn_params["output_c5"] = nn.Conv2d(
            self.out_channels,
            self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    @property
    def fpn_strides(self):
        """
        Total stride up to the FPN level. For a fixed ConvNet, these values
        are invariant to input image size. You may access these values freely
        to implement your logic in FCOS / Faster R-CNN.
        """
        return {"p3": 8, "p4": 16, "p5": 32}

    def forward(self, images: torch.Tensor):
        # Multi-scale features, dictionary with keys: {"c3", "c4", "c5"}.
        backbone_feats = self.backbone(images)

        fpn_feats = {"p3": None, "p4": None, "p5": None}

        c3 = backbone_feats["c3"]
        c4 = backbone_feats["c4"]
        c5 = backbone_feats["c5"]

        c3_conv = self.fpn_params["lateral_c3"](c3)
        c4_conv = self.fpn_params["lateral_c4"](c4)
        c5_conv = self.fpn_params["lateral_c5"](c5)

        c4_conv_2x = F.interpolate(
            c4_conv, scale_factor=2, mode="bilinear", align_corners=False
        )

        c5_conv_2x = F.interpolate(
            c5_conv, scale_factor=2, mode="bilinear", align_corners=False
        )
        c5_conv_4x = F.interpolate(
            c5_conv, scale_factor=4, mode="bilinear", align_corners=False
        )

        p3 = self.fpn_params["output_c3"](c3_conv + c4_conv_2x + c5_conv_4x)
        p4 = self.fpn_params["output_c4"](c4_conv + c5_conv_2x)
        p5 = self.fpn_params["output_c5"](c5_conv)

        fpn_feats["p3"] = p3
        fpn_feats["p4"] = p4
        fpn_feats["p5"] = p5

        return fpn_feats


def get_fpn_location_coords(
    shape_per_fpn_level: Dict[str, Tuple],
    strides_per_fpn_level: Dict[str, int],
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Map every location in FPN feature map to a point on the image. This point
    represents the center of the receptive field of this location. We need to
    do this for having a uniform co-ordinate representation of all the locations
    across FPN levels, and GT boxes.

    Args:
        shape_per_fpn_level: Shape of the FPN feature level, dictionary of keys
            {"p3", "p4", "p5"} and feature shapes `(B, C, H, W)` as values.
        strides_per_fpn_level: Dictionary of same keys as above, each with an
            integer value giving the stride of corresponding FPN level.
            See `backbone.py` for more details.

    Returns:
        Dict[str, torch.Tensor]
            Dictionary with same keys as `shape_per_fpn_level` and values as
            tensors of shape `(H * W, 2)` giving `(xc, yc)` co-ordinates of the
            centers of receptive fields of the FPN locations, on input image.
    """

    # Set these to `(N, 2)` Tensors giving absolute location co-ordinates.
    location_coords = {}

    for level_name, feat_shape in shape_per_fpn_level.items():
        level_stride = strides_per_fpn_level[level_name]

        h, w = feat_shape[2], feat_shape[3]

        x_coords = torch.arange(w)
        y_coords = torch.arange(h)

        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing="ij")

        xc = (grid_x + 0.5) * level_stride
        yc = (grid_y + 0.5) * level_stride

        coords = torch.stack((xc, yc), dim=-1).view(-1, 2)

        location_coords[level_name] = coords.to(device=device, dtype=dtype)

    return location_coords


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5):
    """
    Non-maximum suppression removes overlapping bounding boxes.

    Args:
        boxes: Tensor of shape (N, 4) giving top-left and bottom-right coordinates
            of the bounding boxes to perform NMS on.
        scores: Tensor of shpe (N, ) giving scores for each of the boxes.
        iou_threshold: Discard all overlapping boxes with IoU > iou_threshold

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """

    if (not boxes.numel()) or (not scores.numel()):
        return torch.zeros(0, dtype=torch.long)

    keep = []

    _, indices = torch.sort(scores, descending=True)

    while indices.numel() > 0:
        i = indices[0].item()
        keep.append(i)

        if indices.numel() == 1:
            break

        x1_A, y1_A, x2_A, y2_A = boxes[i]

        remaining = indices[1:]
        x1_B = boxes[remaining, 0]
        y1_B = boxes[remaining, 1]
        x2_B = boxes[remaining, 2]
        y2_B = boxes[remaining, 3]

        x_left = torch.max(x1_A, x1_B)
        y_top = torch.max(y1_A, y1_B)
        x_right = torch.min(x2_A, x2_B)
        y_bottom = torch.min(y2_A, y2_B)

        width = torch.clamp(x_right - x_left, min=0)
        height = torch.clamp(y_bottom - y_top, min=0)

        intersection = width * height

        area_A = (x2_A - x1_A) * (y2_A - y1_A)
        area_B = (x2_B - x1_B) * (y2_B - y1_B)

        union = area_A + area_B - intersection

        iou = intersection / union

        indices = remaining[iou <= iou_threshold]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def class_spec_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    iou_threshold: float = 0.5,
):
    """
    Wrap `nms` to make it class-specific. Pass class IDs as `class_ids`.
    STUDENT: This depends on your `nms` implementation.

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    max_coordinate = boxes.max()

    offsets = class_ids.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))

    boxes_for_nms = boxes + offsets[:, None]

    keep = nms(boxes_for_nms, scores, iou_threshold)

    return keep
