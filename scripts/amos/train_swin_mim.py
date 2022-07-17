from dataclasses import dataclass, field

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
import torch
import wandb

from umei.datasets.amos import AmosSnimDataModule, AmosArgs
from umei.utils import MyWandbLogger, UMeIParser
from umei.swin_mim import SwinMAEArgs, SwinMIM

@dataclass
class AmosSwinMAEArgs(AmosArgs, SwinMAEArgs):
    num_sanity_val_steps: int = field(default=-1)

def main():
    parser = UMeIParser((AmosSwinMAEArgs, ), use_conf=True)
    args: AmosSwinMAEArgs = parser.parse_args_into_dataclasses()[0]
    print(args)
    pl.seed_everything(args.seed)
    datamodule = AmosSnimDataModule(args)
    output_dir = args.output_dir / f'mask{args.mask_ratio * 100}-r{args.non_mask_factor}' / f'run-{args.seed}'
    output_dir.mkdir(exist_ok=True, parents=True)
    trainer = pl.Trainer(
        logger=MyWandbLogger(
            project='amos-swin_mim',
            name=f'{args.exp_name}/mask{args.mask_ratio * 100}-r{args.non_mask_factor}/run-{args.seed}',
            save_dir=str(output_dir),
            group=args.exp_name,
            offline=args.log_offline,
            resume=args.resume_log,
        ),
        callbacks=[
            ModelCheckpoint(
                dirpath=output_dir,
                verbose=True,
                save_last=True,
                save_top_k=0,
                every_n_epochs=1,
                save_on_train_epoch_end=True,
            ),
            LearningRateMonitor(logging_interval='epoch')
        ],
        num_nodes=args.num_nodes,
        gpus=torch.cuda.device_count(),
        precision=args.precision,
        benchmark=True,
        max_epochs=int(args.num_train_epochs),
        log_every_n_steps=5,
        strategy=DDPStrategy(find_unused_parameters=args.ddp_find_unused_parameters),
        num_sanity_val_steps=args.num_sanity_val_steps,
    )
    model = SwinMIM(args)
    last_ckpt_path = output_dir / 'last.ckpt'
    if not last_ckpt_path.exists():
        last_ckpt_path = None
    if trainer.is_global_zero:
        conf_save_path = output_dir / 'conf.yml'
        if conf_save_path.exists():
            conf_save_path.rename(output_dir / 'conf-save.yml')
        UMeIParser.save_args_as_conf(args, conf_save_path)
    trainer.fit(model, datamodule=datamodule, ckpt_path=last_ckpt_path)
    wandb.finish()

if __name__ == '__main__':
    main()
