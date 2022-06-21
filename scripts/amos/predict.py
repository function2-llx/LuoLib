from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.trainer.supporters import CombinedLoader
import torch
from torch import nn
from transformers import HfArgumentParser

import monai
from monai.inferers import sliding_window_inference
from monai.utils import BlendMode, ImageMetaKey
from umei.datasets.amos import AmosArgs, AmosDataModule, AmosModel

@dataclass
class PredictionArgs:
    model_output_dirs: list[Path]
    output_dir: Path
    sw_overlap: float = field(default=0.25)
    overwrite: bool = field(default=False)

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
            resample=False,
            separate_folder=False,
        )

        for output_dir in args.model_output_dirs:
            output_dir = Path(output_dir)
            args: AmosArgs = AmosArgs.from_yaml_file(output_dir / 'conf.yml')
            args.dataloader_num_workers = 0
            self.datamodules.append(AmosDataModule(args))
            self.models.append(AmosModel.load_from_checkpoint(str(output_dir / 'last.ckpt'), args=args))
        if not self.args.overwrite:
            subjects = [
                subject.name[:-sum(map(len, subject.suffixes))]
                for subject in self.args.output_dir.iterdir()
            ]
            for datamodule in self.datamodules:
                datamodule.exclude_test(subjects)

    def predict_dataloader(self):
        return CombinedLoader([datamodule.predict_dataloader() for datamodule in self.datamodules])

    def predict_step(self, combined_batch: list[dict], *args, **kwargs):
        ensemble_logits: Optional[torch.Tensor] = None
        example_batch = combined_batch[0]
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
                sw_batch_size=model.args.sw_batch_size,
                predictor=model.forward,
                overlap=self.args.sw_overlap,
                mode=BlendMode.GAUSSIAN,
                progress=True,
            )[0]
            resampled_pred_logit, _ = self.resampler.__call__(
                pred_logit,
                src_affine=batch['img_meta_dict']['affine'],
                dst_affine=example_meta_dict['original_affine'],
                spatial_size=example_meta_dict['spatial_shape'],
            )
            if ensemble_logits is None:
                ensemble_logits = resampled_pred_logit
            else:
                ensemble_logits += resampled_pred_logit
        ensemble_logits /= len(combined_batch)
        pred = ensemble_logits.argmax(dim=0, keepdim=True)
        self.saver.__call__(pred, meta_data={
            ImageMetaKey.FILENAME_OR_OBJ: example_batch['subject'],
            'affine': example_meta_dict['original_affine'],
        })

def main():
    parser = HfArgumentParser(PredictionArgs)
    args: PredictionArgs = parser.parse_args_into_dataclasses()[0]

    predictor = AmosEnsemblePredictor(args)
    trainer = pl.Trainer(
        logger=False,
        gpus=torch.cuda.device_count(),
        precision=16,
        benchmark=True,
        strategy=DDPStrategy()
    )
    trainer.predict(predictor)

if __name__ == '__main__':
    main()
