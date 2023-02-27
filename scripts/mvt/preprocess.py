import itertools as it
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.contrib.concurrent import process_map

import monai
from monai import transforms as monai_t
from monai.data import MetaTensor

output_root = Path('mvt-data')

class ResizeMax(monai_t.Transform):
    def __init__(self, max_whole_size: int = 512):
        self.max_whole_size = max_whole_size

    def __call__(self, img: MetaTensor):
        spatial_shape = img.shape[1:]
        resize = np.minimum(self.max_whole_size, spatial_shape)
        if spatial_shape != resize:
            resizer = monai_t.Resize(resize, anti_aliasing=True)
            img = resizer(img)
        return img

def process(task: tuple[Path, Path]):
    img_path, output_dir = task
    if output_dir.exists():
        return
    output_dir.mkdir(parents=True)
    loader = monai.transforms.Compose([
        monai_t.LoadImage(image_only=True, ensure_channel_first=True),
        monai_t.Lambda(lambda x: x - x.min()),
        monai_t.CropForeground(),
        monai_t.NormalizeIntensity(nonzero=True, set_zero_to_min=True),
        ResizeMax(),
    ])
    img: MetaTensor = loader(img_path)

    np.save(str(output_dir / 'data.npy'), img[0].numpy())
    np.save(str(output_dir / 'affine.npy'), img.affine)

def main():
    global output_root

    tasks = []
    for splits, splits_name in [
        (['Tr', 'Va'], 'train+val'),
        (['Ts'], 'test'),
    ]:
        output_dir = output_root / 'AMOS22' / splits_name
        # this is the only one that PyCharm could recognize type (it.chain, itz.concat not work)
        for img_path in it.chain.from_iterable([
            (Path('datasets/AMOS22/all') / f'images{split}').glob('*.nii.gz')
            for split in splits
        ]):
            case = img_path.with_suffix('').stem
            case_id = int(case[-4:])
            modality = 'CT' if case_id <= 500 else 'MRI'
            tasks.append((img_path, output_dir / modality / case))

    process_map(process, tasks, max_workers=8, dynamic_ncols=True, desc='AMOS22')

if __name__ == '__main__':
    main()
