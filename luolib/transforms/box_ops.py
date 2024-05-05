from collections.abc import Sequence

import einops
import torch

from monai.config import NdarrayOrTensor, NdarrayTensor
from monai.data.box_utils import TO_REMOVE, get_spatial_dims
from monai.utils import convert_data_type, convert_to_dst_type, convert_to_tensor

assert TO_REMOVE == 0
EPS = 1e-8

# copy these functions to use torch.float64
def _apply_affine_to_points(points: torch.Tensor, affine: torch.Tensor, include_shift: bool = True) -> torch.Tensor:
    """
    This internal function applies affine matrices to the point coordinate

    Args:
        points: point coordinates, Nx2 or Nx3 torch tensor or ndarray, representing [x, y] or [x, y, z]
        affine: affine matrix to be applied to the point coordinates, sized (spatial_dims+1,spatial_dims+1)
        include_shift: default True, whether the function apply translation (shift) in the affine transform

    Returns:
        transformed point coordinates, with same data type as ``points``, does not share memory with ``points``
    """

    spatial_dims = get_spatial_dims(points=points)

    # compute new points
    if include_shift:
        # append 1 to form Nx(spatial_dims+1) vector, then transpose
        points_affine = torch.cat(
            [points, torch.ones(points.shape[0], 1, device=points.device, dtype=points.dtype)], dim=1
        ).transpose(0, 1)
        # apply affine
        points_affine = torch.matmul(affine, points_affine)
        # remove appended 1 and transpose back
        points_affine = points_affine[:spatial_dims, :].transpose(0, 1)
    else:
        points_affine = points.transpose(0, 1)
        points_affine = torch.matmul(affine[:spatial_dims, :spatial_dims], points_affine)
        points_affine = points_affine.transpose(0, 1)

    return points_affine


def apply_affine_to_boxes(boxes: NdarrayTensor, affine: NdarrayOrTensor) -> NdarrayTensor:
    """
    This function applies affine matrices to the boxes

    Args:
        boxes: bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be StandardMode
        affine: affine matrix to be applied to the box coordinates, sized (spatial_dims+1,spatial_dims+1)

    Returns:
        returned affine transformed boxes, with same data type as ``boxes``, does not share memory with ``boxes``
    """

    # convert numpy to tensor if needed
    boxes_t, *_ = convert_data_type(boxes, torch.Tensor)

    # some operation does not support torch.float16
    # convert to float32

    boxes_t = boxes_t.to(dtype=torch.float64)
    affine_t, *_ = convert_to_dst_type(src=affine, dst=boxes_t)

    spatial_dims = get_spatial_dims(boxes=boxes_t)

    # affine transform left top and bottom right points
    # might flipped, thus lt may not be left top any more
    lt: torch.Tensor = _apply_affine_to_points(boxes_t[:, :spatial_dims], affine_t, include_shift=True)
    rb: torch.Tensor = _apply_affine_to_points(boxes_t[:, spatial_dims:], affine_t, include_shift=True)

    # make sure lt_new is left top, and rb_new is bottom right
    lt_new, _ = torch.min(torch.stack([lt, rb], dim=2), dim=2)
    rb_new, _ = torch.max(torch.stack([lt, rb], dim=2), dim=2)

    boxes_t_affine = torch.cat([lt_new, rb_new], dim=1)

    # convert tensor back to numpy if needed
    boxes_affine: NdarrayOrTensor
    boxes_affine, *_ = convert_to_dst_type(src=boxes_t_affine, dst=boxes)
    return boxes_affine  # type: ignore[return-value]

def apply_affine_to_boxes_int(boxes: NdarrayTensor, affine: NdarrayOrTensor) -> NdarrayTensor:
    boxes_t: torch.Tensor = convert_to_tensor(boxes).clone()
    d = get_spatial_dims(boxes)
    boxes_t[:, d:] -= 1
    boxes_f = apply_affine_to_boxes(boxes_t.double(), affine)
    boxes_f[:, d:] += 1
    boxes_t = round_boxes(boxes_f)
    boxes, *_ = convert_to_dst_type(boxes_t, boxes, dtype=torch.int64)
    return boxes

def round_boxes(boxes: NdarrayTensor) -> NdarrayTensor:
    boxes_t = convert_to_tensor(boxes)
    boxes_int = torch.empty_like(boxes_t, dtype=torch.int64)
    d = get_spatial_dims(boxes)
    boxes_int[:, :d] = (boxes_t[:, :d] + EPS).floor()
    boxes_int[:, d:] = (boxes_t[:, d:] - EPS).ceil()
    boxes_int, *_ = convert_to_dst_type(boxes_int, boxes, dtype=torch.int64)
    return boxes_int

def norm_boxes(boxes: NdarrayTensor, norm_size: Sequence[int]) -> NdarrayTensor:
    boxes_t = convert_to_tensor(boxes)
    norm_size_t = einops.repeat(torch.tensor(norm_size), 'd -> (l2 d)', l2=2)
    boxes_t = boxes_t.double() / norm_size_t
    boxes, *_ = convert_to_dst_type(boxes_t, boxes, dtype=torch.float64)
    return boxes
