from pathlib import Path

from PIL import Image
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch

import monai
from monai import transforms as monai_t
from monai.utils import NumpyPadMode
from umei.snim.model_omega import SnimModel
from umei.snim.omega import SnimConf
from umei.utils import DataKey

plot_dir = Path('snim-plot')

def save_slice(img: torch.Tensor, slice_idx: int, path: Path, min_v: float = None, max_v: float = None):
    if min_v is None:
        min_v = img[..., slice_idx].min().item()
    if max_v is None:
        max_v = img[..., slice_idx].max().item()
    img = img.clamp(min_v, max_v)
    img = (img - min_v) / (max_v - min_v)
    path.parent.mkdir(exist_ok=True, parents=True)
    img_slice: np.ndarray = img[0, :, :, slice_idx].rot90(dims=(0, 1)).cpu().numpy()
    img_slice *= 255
    Image.fromarray(img_slice.astype(np.uint8)).save(path)

def main():
    monai.utils.set_determinism(888)
    torch.set_float32_matmul_precision('high')
    torch.set_grad_enabled(False)
    pt_setting = 'snim/mask80-16x16x4-act1k+amos/run-42'
    output_dir = Path('output-omega') / pt_setting
    conf = OmegaConf.load(output_dir / 'conf.yaml')
    conf.merge_with(OmegaConf.structured(SnimConf))
    conf = OmegaConf.to_object(conf)

    transform = monai.transforms.Compose([
            monai_t.LoadImage(image_only=True, ensure_channel_first=True),
            monai_t.SpatialPad(
                spatial_size=conf.sample_shape,
                mode=NumpyPadMode.CONSTANT,
                pad_min=True,
            ),
            monai_t.RandSpatialCrop(
                roi_size=conf.sample_shape,
                random_size=False,
            ),
        ]
    )

    img_paths = [
        ('act1k', Path('snim-data/act1k/Case_00701_0000.npy')),
        ('amos', Path('snim-data/amos/amos_0497.npy')),
        ('btcv', Path('snim-data/btcv/img0075.npy')),
    ]

    img = torch.stack([
        transform(img_path)
        for _, img_path in img_paths
    ]).cuda()
    slice_idxes = [12, 15, 18, 24, 30, 36, 42, 45]
    for i in range(img.shape[0]):
        for slice_idx in slice_idxes:
            save_slice(img[i], slice_idx, plot_dir / img_paths[i][0] / f'{slice_idx}.png')

    # output_dir = Path('output/snim/ct') / pt_setting / 'amos/run-42'
    # args: SnimArgs = SnimArgs.from_yaml_file(output_dir / 'conf.yml')
    model: SnimModel = SnimModel.load_from_checkpoint(
        output_dir / 'last.ckpt',
        strict=False,
        conf=conf,
    ).cuda().eval()
    mask = model.gen_patch_mask(img.shape[0], img.shape[2:])
    img_mask, pred = model.forward(img, mask, apply_dis_overlay=True)
    # img_mask.clamp_(min=0, max=1)
    # pred.clamp_(min=0, max=1)

    for i in range(img.shape[0]):
        for slice_idx in slice_idxes:
            min_v = img[i, ..., slice_idx].min().item()
            max_v = img[i, ..., slice_idx].max().item()
            save_slice(img_mask[i], slice_idx, plot_dir / img_paths[i][0] / pt_setting / f'mask-{slice_idx}.png', min_v, max_v)
            save_slice(pred[i], slice_idx, plot_dir / img_paths[i][0] / pt_setting / f're-{slice_idx}.png', min_v, max_v)

if __name__ == '__main__':
    with torch.no_grad():
        main()
