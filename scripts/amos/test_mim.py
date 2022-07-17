from dataclasses import dataclass, field

import numpy as np
import pytorch_lightning as pl
import wandb

from umei.datasets.amos import AmosArgs, AmosSnimDataModule
from umei.swin_mim import SwinMIM, SwinMAEArgs
from umei.utils import MyWandbLogger, UMeIParser

@dataclass
class AmosSwinMAEArgs(AmosArgs, SwinMAEArgs):
    num_sanity_val_steps: int = field(default=-1)

def main():
    parser = UMeIParser((AmosSwinMAEArgs, ), use_conf=True)
    args: AmosSwinMAEArgs = parser.parse_args_into_dataclasses()[0]
    print(args)
    pl.seed_everything(args.seed)
    datamodule = AmosSnimDataModule(args)
    output_dir = args.output_dir / f'mask-{args.mask_ratio * 100}' / f'run-{args.seed}'
    print(f'use output directory: {output_dir}')
    log_save_dir = output_dir / 'mim-test'
    log_save_dir.mkdir(exist_ok=True)
    # output_dir.mkdir(exist_ok=True, parents=True)
    trainer = pl.Trainer(
        logger=MyWandbLogger(
            project='amos-swin_mim-test',
            name=f'{args.exp_name}/mask-{args.mask_ratio * 100}/run-{args.seed}',
            save_dir=str(log_save_dir),
            group=args.exp_name,
            offline=args.log_offline,
            resume=args.resume_log,
        ),
        gpus=1,
        precision=args.precision,
        benchmark=True,
    )
    last_ckpt_path = output_dir / 'last.ckpt'
    for mask_ratio in np.linspace(0, 1, 11):
        args.mask_ratio = mask_ratio
        model = SwinMIM(args)
        trainer.validate(model, datamodule=datamodule, ckpt_path=str(last_ckpt_path))
    wandb.finish()

if __name__ == '__main__':
    main()
