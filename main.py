from typing import Optional

import numpy as np
from pytorch_lightning import LightningDataModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torch

from monai.data import Dataset

from umei.argparse import UMeIParser
from umei.args import UMeIArgs
from umei.umei import UMeI
from umei.model import UEncoderBase

def build_encoder(args: UMeIArgs) -> UEncoderBase:
    from monai.networks.nets import resnet18
    return resnet18(n_input_channels=args.num_input_channels, feed_forward=False)

# cannot wait: https://github.com/PyTorchLightning/pytorch-lightning/pull/12172/
class MyWandbLogger(WandbLogger):
    @WandbLogger.name.getter
    def name(self) -> Optional[str]:
        return self._experiment.name if self._experiment else self._name

def get_datamodule(args: UMeIArgs) -> LightningDataModule:
    # TODO: load data
    return LightningDataModule.from_datasets(
        train_dataset=Dataset([
            {
                args.img_key: torch.randn(1, 64, 64, 16),
                args.cls_key: np.random.randint(args.num_cls_classes),
            }
            for _ in range(20)
        ]),
        val_dataset=Dataset([
            {
                args.img_key: torch.randn(1, 64, 64, 16),
                args.cls_key: np.random.randint(args.num_cls_classes),
            }
            for _ in range(5)
        ]),
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

def main():
    parser = UMeIParser((UMeIArgs, ), use_conf=True)
    args: UMeIArgs = parser.parse_args_into_dataclasses()[0]
    umei_model = UMeI(args, encoder=build_encoder(args))
    trainer = pl.Trainer(
        logger=MyWandbLogger(
            name=args.exp_name,
            save_dir=str(args.output_dir),
        ) if args.log else None,
        gpus=args.n_gpu,
        precision=args.precision,
        benchmark=True,
        max_epochs=int(args.num_train_epochs),
        callbacks=[
            ModelCheckpoint(dirpath=args.output_dir, monitor=args.monitor, save_last=True),
        ],
        num_sanity_val_steps=0,
        strategy=None,
    )
    datamodule = get_datamodule(args)
    trainer.fit(umei_model, datamodule=datamodule)

if __name__ == '__main__':
    main()
