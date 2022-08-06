from dataclasses import dataclass, field

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
import torch
import wandb

from umei.snim import SnimArgs, SnimDataModule, SnimModel, build_pretrain_data
from umei.utils import MyWandbLogger, UMeIParser

@dataclass
class TestArgs(SnimArgs):
    test_mask_block_shape: list[int] = field(default=None)

def main():
    parser = UMeIParser((TestArgs, ), use_conf=True)
    args: TestArgs = parser.parse_args_into_dataclasses()[0]
    print(args)
    pl.seed_everything(args.seed)
    exp_suffix = f'mask{args.mask_ratio * 100}-{"x".join(map(str, args.mask_block_shape))}-vf{args.visible_factor}' \
                 f'/run-{args.seed}'
    test_suffix = f'test-{"x".join(map(str, args.test_mask_block_shape))}'

    args.mask_block_shape = args.test_mask_block_shape
    datamodule = SnimDataModule(args, build_pretrain_data(args))
    output_dir = args.output_dir / exp_suffix
    print(f'use output directory: {output_dir}')
    log_save_dir = output_dir / 'snim-test' / test_suffix
    log_save_dir.mkdir(exist_ok=True, parents=True)
    trainer = pl.Trainer(
        logger=MyWandbLogger(
            project='snim-test',
            name=f'{args.exp_name}/{exp_suffix}/{test_suffix}',
            save_dir=str(log_save_dir),
            group=args.exp_name,
            offline=args.log_offline,
            resume=args.resume_log,
        ),
        accelerator='gpu',
        devices=1,
        precision=args.precision,
        benchmark=True,
        strategy=DDPStrategy(),  # compatible
    )
    last_ckpt_path = output_dir / 'last.ckpt'
    for mask_ratio in np.linspace(0, 1, 11):
        args.mask_ratio = mask_ratio
        model = SnimModel(args)
        state_dict: dict = torch.load(last_ckpt_path)['state_dict']
        state_dict.pop('corner_counter.weight')
        model.load_state_dict(state_dict, strict=False)
        trainer.validate(model, datamodule=datamodule)
        # trainer.validate(model, datamodule=datamodule, ckpt_path=str(last_ckpt_path))
    wandb.finish()

if __name__ == '__main__':
    main()
