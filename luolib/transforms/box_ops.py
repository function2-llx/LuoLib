from collections.abc import Sequence

import einops
import torch

from monai.apps.detection.transforms.box_ops import apply_affine_to_boxes
from monai.config import NdarrayOrTensor, NdarrayTensor
from monai.utils import convert_to_dst_type, convert_to_tensor

def convert_boxes_to_int(boxes: NdarrayTensor) -> NdarrayTensor:
    boxes_t = convert_to_tensor(boxes)
    boxes_int = torch.empty_like(boxes_t, dtype=torch.int64)
    boxes_int[:, :3] = boxes_t[:, :3].floor()
    boxes_int[:, 3:] = boxes_t[:, 3:].ceil()
    boxes_int, *_ = convert_to_dst_type(boxes_int, boxes, dtype=torch.int64)
    return boxes_int

def apply_affine_to_boxes_int(boxes: NdarrayTensor, affine: NdarrayOrTensor) -> NdarrayTensor:
    boxes_f = apply_affine_to_boxes(boxes, affine)
    boxes = convert_boxes_to_int(boxes_f)
    return boxes

def norm_boxes(boxes: NdarrayTensor, norm_size: Sequence[int]):
    boxes_t = convert_to_tensor(boxes)
    norm_size_t = einops.repeat(torch.tensor(norm_size), 'd -> (l2 d)', l2=2)
    boxes_t = boxes_t / norm_size_t
    boxes = convert_to_dst_type(boxes_t, boxes)
    return boxes
