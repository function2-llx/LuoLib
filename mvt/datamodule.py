from __future__ import annotations

from functools import cached_property
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Sampler
from tqdm import tqdm

import monai
from monai import transforms as monai_t
from monai.data import CacheDataset, DataLoader, MetaTensor, affine_to_spacing, partition_dataset
from monai.transforms import LoadImage
from monai.utils import GridSamplePadMode
from .args import MVTArgs

class MetaKeys(monai.utils.MetaKeys):
    PATCH_SIZE = "patch_size"

class MVTLoader(monai_t.Transform):
    def __init__(self):
        self.reader = monai.data.NumpyReader()

    def __call__(self, data: dict):
        path = data['path']
        meta = data['meta']
        loader = LoadImage(image_only=True, ensure_channel_first=True)
        img: MetaTensor = loader(path, self.reader)
        img.meta.update(meta)
        return img

class RandCropDynamicPatchSize(monai_t.RandomizableTrait):
    def __init__(self, pad_ratio: float):
        super().__init__()
        self.pad_ratio = pad_ratio

    def __call__(self, img: MetaTensor):
        patch_size = img.meta[MetaKeys.PATCH_SIZE]
        pad_size = (patch_size * self.pad_ratio).astype(int)
        transform = monai_t.Compose([
            monai_t.BorderPad(pad_size, value=img.min()),
            monai_t.SpatialPad(patch_size, value=img.min()),
            monai_t.RandSpatialCrop(patch_size + pad_size, random_center=True, random_size=False),
        ])
        img = transform(img)
        return img

class CenterSpatialCropByKey(monai_t.Transform):
    def __call__(self, img: MetaTensor):
        patch_size = img.meta[MetaKeys.PATCH_SIZE]
        crop = monai_t.CenterSpatialCrop(patch_size)
        img = crop(img)
        plt.imshow(np.rot90(img[0, :, :, img.shape[-1] >> 1]), cmap='gray')
        plt.show()
        return img

class MVTBatchSampler(Sampler):
    def __init__(self, data_source: list[dict], batch_size: int):
        super().__init__(data_source)
        self.batch_size = batch_size
        # group by patch size
        groups = {}
        for i, data in enumerate(data_source):
            groups.setdefault(tuple(data['meta'][MetaKeys.PATCH_SIZE]), []).append(i)
        self.groups = list(map(np.array, groups.values()))
        self.p = np.array(list(map(len, self.groups)), np.float64)
        self.p /= self.p.sum()

    def __iter__(self):
        return self

    @cached_property
    def num_groups(self):
        return len(self.groups)

    def __next__(self):
        group_idx = np.random.choice(range(self.num_groups), p=self.p)
        return np.random.choice(self.groups[group_idx], self.batch_size, replace=True)

class MVTDataModule(pl.LightningDataModule):
    def __init__(self, args: MVTArgs):
        super().__init__()
        self.args = args
        train_data, val_data = [], []
        for dataset_path in args.datasets.paths:
            dataset = []
            for data_path in (args.datasets.root / dataset_path).rglob('data.npy'):
                dataset.append(data_path.parent)
            train_split, val_split = partition_dataset(dataset, ratios=[95, 5], shuffle=True, seed=args.seed)
            train_data.extend(train_split)
            val_data.extend(val_split)
        self.train_data = list(map(self.preprocess_meta, tqdm(train_data, desc='processing train meta')))
        self.val_data = list(map(self.preprocess_meta, tqdm(val_data, desc='processing val meta')))

    def preprocess_meta(self, data_dir: Path):
        args = self.args
        affine: np.ndarray = np.load(str(data_dir / 'affine.npy'))
        spacing: np.ndarray = affine_to_spacing(affine)
        spacing_ratio = np.array(tuple(map(lambda x: 1 << int(x).bit_length() - 1, spacing / spacing.min())))
        patch_size = args.train.max_patch_size // spacing_ratio
        shape = np.load(str(path := data_dir / 'data.npy'), mmap_mode='r').shape
        for i in range(3):
            while patch_size[i] // 2 >= shape[i]:
                patch_size[i] //= 2
        return {
            'path': path,
            'meta': {
                MetaKeys.AFFINE: affine,
                MetaKeys.PATCH_SIZE: patch_size,
            },
        }

    def train_dataloader(self):
        args = self.args
        train_set = CacheDataset(
            self.train_data,
            self.train_transform(),
            args.runtime.num_train_cache,
            num_workers=args.runtime.num_cache_workers,
        )

        return DataLoader(
            train_set,
            batch_sampler=MVTBatchSampler(self.train_data, args.train.train_batch_size),
            num_workers=args.runtime.dataloader_num_workers,
            pin_memory=args.runtime.dataloader_pin_memory,
        )

    def train_transform(self):
        args = self.args
        return monai_t.Compose([
            MVTLoader(),
            RandCropDynamicPatchSize(args.train.pre_pad_ratio),
            monai_t.RandAffine(
                prob=1.,
                rotate_range=args.train.rotate_range,
                rotate_prob=args.train.rotate_prob,
                scale_range=args.train.scale_range,
                scale_prob=args.train.scale_prob,
                padding_mode=GridSamplePadMode.ZEROS,
            ),
            CenterSpatialCropByKey(),
            # monai_t.RandGaussianNoise(prob=args.train.gaussian_noise_prob, std=0.1),
            # monai_t.RandGaussianSmooth(
            #     sigma_x=(0.25, 1.5),
            #     sigma_y=(0.25, 1.5),
            #     sigma_z=(0.25, 1.5),
            #     prob=args.train.gaussian_smooth_prob,
            # ),
            # monai_t.RandLambda(
            #     monai_t.TorchVision("ColorJitter", brightness=tuple(args.train.brightness_range)),
            #     prob=args.train.brightness_prob,
            # ),
            # monai_t.RandLambda(
            #     monai_t.TorchVision("ColorJitter", contrast=tuple(args.train.contrast_range)),
            #     prob=args.train.contrast_prob,
            # ),
            # I think this is a wrong name for gamma correction
            # https://github.com/Project-MONAI/MONAI/discussions/6027
            # monai_t.RandAdjustContrast(
            #     prob=args.train.gamma_correction_prob,
            #     gamma=args.train.gamma_correction_range,
            #     invert_prob=args.train.invert_gamma_correction_prob,
            # ),
            *[
                monai_t.RandFlip(prob=0.5, spatial_axis=i)
                for i in [0, 1, 2]
            ],
        ])
