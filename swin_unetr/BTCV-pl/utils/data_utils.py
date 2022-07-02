# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from monai import data, transforms
from monai.data import load_decathlon_datalist

def get_loader(args):
    data_dir = args.data_dir
    datalist_json = os.path.join(data_dir, args.json_list)

    def fix_seg_affine(data: dict):
        data['label_meta_dict']['affine'] = data[f'image_meta_dict']['affine']
        return data

    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.Lambda(fix_seg_affine),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"],
                                    axcodes="RAS"),
            transforms.Spacingd(keys=["image", "label"],
                                pixdim=(args.space_x, args.space_y, args.space_z),
                                mode=("bilinear", "nearest")),
            transforms.ScaleIntensityRanged(keys=["image"],
                                            a_min=args.a_min,
                                            a_max=args.a_max,
                                            b_min=args.b_min,
                                            b_max=args.b_max,
                                            clip=True),
            # transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.SpatialPadD(
                keys=['image', 'label'],
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
            ),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            transforms.RandFlipd(keys=["image", "label"],
                                 prob=args.RandFlipd_prob,
                                 spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"],
                                 prob=args.RandFlipd_prob,
                                 spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"],
                                 prob=args.RandFlipd_prob,
                                 spatial_axis=2),
            transforms.RandRotate90d(
                keys=["image", "label"],
                prob=args.RandRotate90d_prob,
                max_k=3,
            ),
            transforms.RandScaleIntensityd(keys="image",
                                           factors=0.1,
                                           prob=args.RandScaleIntensityd_prob),
            transforms.RandShiftIntensityd(keys="image",
                                           offsets=0.1,
                                           prob=args.RandShiftIntensityd_prob),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.Lambda(fix_seg_affine),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"],
                                    axcodes="RAS"),
            transforms.Spacingd(keys=["image", "label"],
                                pixdim=(args.space_x, args.space_y, args.space_z),
                                mode=("bilinear", "nearest")),
            transforms.ScaleIntensityRanged(keys=["image"],
                                            a_min=args.a_min,
                                            a_max=args.a_max,
                                            b_min=args.b_min,
                                            b_max=args.b_max,
                                            clip=True),
            # transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            # transforms.Orientationd(keys=["image"], axcodes="RAS"),
            transforms.Spacingd(keys="image",
                                pixdim=(args.space_x, args.space_y, args.space_z),
                                mode="bilinear"),
            transforms.ScaleIntensityRanged(keys=["image"],
                                            a_min=args.a_min,
                                            a_max=args.a_max,
                                            b_min=args.b_min,
                                            b_max=args.b_max,
                                            clip=True),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    if args.test_mode:
        test_files = load_decathlon_datalist(datalist_json,
                                            True,
                                            "validation",
                                            base_dir=data_dir)
        test_ds = data.Dataset(data=test_files, transform=test_transform)
        test_loader = data.DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=True,
        )
        loader = test_loader
    else:
        datalist = load_decathlon_datalist(datalist_json,
                                           True,
                                           "training",
                                           base_dir=data_dir)
        if args.use_normal_dataset:
            train_ds = data.Dataset(data=datalist, transform=train_transform)
        else:
            train_ds = data.CacheDataset(
                data=datalist,
                transform=train_transform,
                cache_num=80,
                cache_rate=1.0,
                num_workers=args.workers,
            )
        train_loader = data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
        )
        val_files = load_decathlon_datalist(datalist_json,
                                            True,
                                            "validation",
                                            base_dir=data_dir)
        val_ds = data.Dataset(data=val_files, transform=val_transform)
        val_loader = data.DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )
        loader = [train_loader, val_loader]

    return loader
