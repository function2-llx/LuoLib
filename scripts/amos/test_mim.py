from dataclasses import dataclass, field

import pytorch_lightning as pl
import wandb

from umei.datasets.amos import AmosArgs, AmosSwinMAEDataModule
from umei.swin_mae import SwinMAE, SwinMAEArgs
from umei.utils import MyWandbLogger, UMeIParser

@dataclass
class AmosSwinMAEArgs(AmosArgs, SwinMAEArgs):
    num_sanity_val_steps: int = field(default=-1)

def main():
    parser = UMeIParser((AmosSwinMAEArgs, ), use_conf=True)
    args: AmosSwinMAEArgs = parser.parse_args_into_dataclasses()[0]
    print(args)
    pl.seed_everything(args.seed)
    datamodule = AmosSwinMAEDataModule(args)
    output_dir = args.output_dir / f'mask-{args.mask_ratio * 100}' / f'run-{args.seed}'
    # output_dir.mkdir(exist_ok=True, parents=True)
    trainer = pl.Trainer(
        logger=MyWandbLogger(
            project='amos-swin_mae-test',
            name=f'{args.exp_name}/mask-{args.mask_ratio * 100}/run-{args.seed}',
            save_dir=str(output_dir),
            group=args.exp_name,
            offline=args.log_offline,
            resume=args.resume_log,
        ),
        gpus=1,
        precision=args.precision,
        benchmark=True,
    )
    last_ckpt_path = output_dir / 'last.ckpt'
    for mask_ratio in [0, 0.2, 0.4, 0.6, 1]:
        args.mask_ratio = mask_ratio
        model = SwinMAE(args)
        trainer.validate(model, datamodule=datamodule, ckpt_path=str(last_ckpt_path))
    wandb.finish()

if __name__ == '__main__':
    main()
