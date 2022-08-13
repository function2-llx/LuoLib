from pathlib import Path

from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import torch

import monai
from monai.utils import GridSampleMode, NumpyPadMode
from umei.datasets.amos import AMOS_DATA_DIR
from umei.datasets.btcv import BTCV_DATA_DIR
from umei.datasets.act1k import ACT1K_DATA_DIR
from umei.snim import SnimArgs, SnimModel
from umei.utils import DataKey

plot_dir = Path('snim-plot')

def save_slice(img: torch.Tensor, slice_idx: int, path: Path):
    path.parent.mkdir(exist_ok=True, parents=True)
    img_slice: np.ndarray = img[0, :, :, slice_idx].rot90(dims=(0, 1)).cpu().numpy()
    img_slice *= 255
    Image.fromarray(img_slice.astype(np.uint8)).save(path)

def main():
    loader = monai.transforms.Compose([
            monai.transforms.LoadImageD(DataKey.IMG),
            monai.transforms.AddChannelD(DataKey.IMG),
            monai.transforms.OrientationD(DataKey.IMG, axcodes='RAS'),
            monai.transforms.SpacingD(DataKey.IMG, pixdim=(1.5, 1.5, 2), mode=GridSampleMode.BILINEAR),
            monai.transforms.ScaleIntensityRangeD(
                DataKey.IMG,
                a_min=-175,
                a_max=250,
                b_min=0,
                b_max=1,
                clip=True,
            ),
            monai.transforms.CropForegroundD(DataKey.IMG, source_key=DataKey.IMG),
            monai.transforms.SpatialPadD(
                DataKey.IMG,
                spatial_size=(96, 96, 96),
                mode=NumpyPadMode.CONSTANT,
            ),
            monai.transforms.CenterSpatialCropD(DataKey.IMG, roi_size=(96, 96, 96)),
            monai.transforms.Lambda(lambda data: data[DataKey.IMG].as_tensor().cuda()),
        ])

    img_paths = [
        ('amos', AMOS_DATA_DIR / 'imagesTs/amos_0022.nii.gz'),
        ('btcv', BTCV_DATA_DIR / 'imagesTs/img0070.nii.gz'),
        ('act1k', ACT1K_DATA_DIR / 'Image/Case_00942_0000.nii.gz'),
    ]

    img = torch.stack([
        loader({DataKey.IMG: img_path})
        for _, img_path in img_paths
    ])
    slice_idxes = [31, 63]
    for i in range(img.shape[0]):
        for slice_idx in slice_idxes:
            save_slice(img[i], slice_idx, plot_dir / img_paths[i][0] / f'{slice_idx}.png')

    pt_settings = [
        'dist-b-96x96/mask85.0-2x2x2-vf0.1',
        'param-b-96x96/mask85.0-2x2x2-vf0.0',
        'dist-b-96x96/mask85.0-16x16x16-vf0.1',
        'param-b-96x96/mask85.0-16x16x16-vf0.0',
    ]
    mask = {}
    for pt_setting in pt_settings:
        output_dir = Path('output/snim/ct') / pt_setting / 'amos/run-42'
        args: SnimArgs = SnimArgs.from_yaml_file(output_dir / 'conf.yml')
        model: SnimModel = SnimModel.load_from_checkpoint(
            str(output_dir / 'last.ckpt'),
            strict=False,
            args=args,
        ).cuda().eval()
        if tuple(args.mask_block_shape) not in mask:
            mask[tuple(args.mask_block_shape)] = model.gen_patch_mask(img.shape[0], img.shape[2:])
        _, _, _, _, img_mask, pred = model.forward(img, mask[tuple(args.mask_block_shape)])
        img_mask.clamp_(min=0, max=1)
        pred.clamp_(min=0, max=1)
        for i in range(img.shape[0]):
            for slice_idx in slice_idxes:
                save_slice(img_mask[i], slice_idx, plot_dir / img_paths[i][0] / pt_setting / f'mask-{slice_idx}.png')
                save_slice(pred[i], slice_idx, plot_dir / img_paths[i][0] / pt_setting / f're-{slice_idx}.png')

if __name__ == '__main__':
    with torch.no_grad():
        main()
