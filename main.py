from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torch

from umei import UEncoderBase, UMeI
from umei.argparse import UMeIParser
from umei.datasets.stoic2021.datamodule import Stoic2021DataModule
from umei.utils.args import UMeIArgs

torch.multiprocessing.set_sharing_strategy('file_system')

def build_encoder(args: UMeIArgs) -> UEncoderBase:
    from monai.networks.nets import resnet18
    return resnet18(n_input_channels=args.num_input_channels, feed_forward=False)

# cannot wait: https://github.com/PyTorchLightning/pytorch-lightning/pull/12172/
class MyWandbLogger(WandbLogger):
    @WandbLogger.name.getter
    def name(self) -> Optional[str]:
        return self._experiment.name if self._experiment else self._name

def main():
    parser = UMeIParser((UMeIArgs, ), use_conf=True)
    args: UMeIArgs = parser.parse_args_into_dataclasses()[0]
    datamodule = Stoic2021DataModule(args)
    umei_model = UMeI(args, encoder=build_encoder(args))
    trainer = pl.Trainer(
        logger=MyWandbLogger(
            name=args.exp_name,
            save_dir=str(args.output_dir),
        ) if args.log else None,
        gpus=1,
        precision=args.precision,
        benchmark=True,
        max_epochs=int(args.num_train_epochs),
        callbacks=[
            ModelCheckpoint(dirpath=args.output_dir, monitor=args.monitor, save_last=True),
        ],
        num_sanity_val_steps=0,
        strategy=None,
    )
    trainer.fit(umei_model, datamodule=datamodule)

if __name__ == '__main__':
    main()
