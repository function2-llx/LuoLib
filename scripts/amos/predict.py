from __future__ import annotations

from dataclasses import dataclass, field
import itertools
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.trainer.supporters import CombinedLoader
import torch
from torch import nn
from tqdm import tqdm

import monai
from monai.data import NibabelWriter
from monai.inferers import sliding_window_inference
from monai.utils import BlendMode, ImageMetaKey
from umei.datasets.amos import AmosArgs, AmosDataModule, AmosModel
from umei.utils import UMeIParser

@dataclass
class PredictionArgs:
    model_output_dirs: list[Path]
    output_dir: Path
    sw_overlap: float = field(default=0.25)
    sw_batch_size: int = field(default=16)
    overwrite: bool = field(default=False)
    different_dataloader: bool = field(default=False)
    il: int = field(default=None)
    ir: int = field(default=None)
    post_labels: list[int] = field(default_factory=list)

class AmosEnsemblePredictor(pl.LightningModule):
    def __init__(self, args: PredictionArgs):
        super().__init__()
        self.args = args
        self.datamodules = []
        self.models = nn.ModuleList()
        self.resampler = monai.transforms.SpatialResample()
        self.saver = monai.transforms.SaveImage(
            output_dir=self.args.output_dir,
            output_postfix='',
            output_dtype=np.uint8,
            resample=False,
            separate_folder=False,
            writer=NibabelWriter,   # type: ignore
        )
        self.post_transform = monai.transforms.Compose([
            monai.transforms.KeepLargestConnectedComponent(is_onehot=False, applied_labels=args.post_labels),
        ])

        predicted_subjects = []
        if not args.overwrite:
            predicted_subjects = [
                subject.name[:-sum(map(len, subject.suffixes))]
                for subject in args.output_dir.iterdir()
            ]

        for i, output_dir in enumerate(tqdm(args.model_output_dirs, ncols=80, desc='loading models')):
            output_dir = Path(output_dir)
            model_args: AmosArgs = AmosArgs.from_yaml_file(output_dir / 'conf.yml')
            datamodule = AmosDataModule(model_args)
            datamodule.filter_test(predicted_subjects, idx_start=args.il, idx_end=args.ir, print_included=(i == 0))
            self.datamodules.append(datamodule)
            self.models.append(AmosModel.load_from_checkpoint(str(output_dir / 'last.ckpt'), args=model_args))

    def predict_dataloader(self):
        if self.args.different_dataloader:
            return CombinedLoader([datamodule.predict_dataloader() for datamodule in self.datamodules])
        else:
            return self.datamodules[0].predict_dataloader()

    def predict_step(self, combined_batch: list[dict], *args, **kwargs):
        ensemble_dist: Optional[torch.Tensor] = None
        if self.args.different_dataloader:
            assert isinstance(combined_batch, list)
            example_batch = combined_batch[0]
        else:
            example_batch = combined_batch
            combined_batch = itertools.repeat(combined_batch)
        subject = example_batch['subject']

        if (self.args.output_dir / f'{subject}.nii.gz').exists() and not self.args.overwrite:
            print(f'skip {subject}')
            return

        print(f'predicting {subject}')
        example_meta_dict = example_batch['img_meta_dict']
        for batch, model in zip(combined_batch, self.models):
            model: AmosModel
            pred_logit = sliding_window_inference(
                batch['img'],
                roi_size=model.args.sample_shape,
                sw_batch_size=self.args.sw_batch_size,
                predictor=model.forward,
                overlap=self.args.sw_overlap,
                mode=BlendMode.GAUSSIAN,
                device='cpu',   # save gpu memory -- which can be a lot!
                progress=True,
            )[0]
            resampled_pred_logit = self.resampler.__call__(
                pred_logit,
                src_affine=batch['img_meta_dict']['affine'],
                dst_affine=example_meta_dict['original_affine'],
                spatial_size=example_meta_dict['spatial_shape'],
            )
            pred_dist = resampled_pred_logit.softmax(dim=0)
            if ensemble_dist is None:
                ensemble_dist = pred_dist
            else:
                ensemble_dist += pred_dist
        ensemble_dist /= len(self.models)
        ensemble_pred = ensemble_dist.argmax(dim=0, keepdim=True)
        pred = self.post_transform(ensemble_pred)
        # using the original header to pass the "tolerance" check for MRI
        # currently monai does not preserve the exact header
        nib.save(
            nib.Nifti1Image(
                pred[0].cpu().numpy().astype(np.uint8),
                affine=example_meta_dict['original_affine'],
                header=nib.load(example_meta_dict[ImageMetaKey.FILENAME_OR_OBJ]).header,
            ),
            self.args.output_dir / f"{example_batch['subject']}.nii.gz",
        )

def main():
    parser = UMeIParser(PredictionArgs, use_conf=True, infer_output=False)
    args: PredictionArgs = parser.parse_args_into_dataclasses()[0]
    print(args)
    args.output_dir.mkdir(exist_ok=True, parents=True)

    predictor = AmosEnsemblePredictor(args)
    trainer = pl.Trainer(
        logger=False,
        gpus=torch.cuda.device_count(),
        precision=16,
        benchmark=True,
        strategy=DDPStrategy(),
    )
    trainer.predict(predictor)

if __name__ == '__main__':
    main()
