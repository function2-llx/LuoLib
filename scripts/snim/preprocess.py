from pathlib import Path
import itertools as it
from toolz import itertoolz as itz

import numpy as np
from tqdm.contrib.concurrent import process_map

import monai
from monai import transforms as monai_t
from monai.data import ImageWriter

output_dir = Path('snim-data')

loader = monai_t.Compose([
    monai_t.LoadImage(image_only=True, ensure_channel_first=True),
    monai_t.CropForeground('min'),
    monai_t.NormalizeIntensity(non_min=True),
    monai_t.Spacing([0.75, 0.75, 3]),
])

def process(path: Path, dataset: str):
    img = loader(path)
    case = path.with_suffix('').stem
    save_path = output_dir / dataset / f'{case}.npy'
    save_path.parent.mkdir(exist_ok=True, parents=True)
    np.save(save_path, img[0].numpy())

def get_amos():
    paths = []
    for path in Path('datasets/AMOS22/all').glob('images*/*.nii.gz'):
        case = path.with_suffix('').stem
        case_id = int(case[-4:])
        if case_id <= 500:
            paths.append(path)
    return paths

def main():
    max_workers = 8
    process_map(process, get_amos(), it.repeat('amos'), max_workers=max_workers)
    process_map(process, list(Path('datasets/AbdomenCT-1K/Image').glob('*.nii.gz')), it.repeat('act1k'), max_workers=max_workers)

if __name__ == '__main__':
    main()
