from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary
from pytorch_lightning.strategies import DDPStrategy
import torch
import wandb

from umei.datasets.btcv import BTCVArgs, BTCVDataModule
from umei.datasets.btcv.model import BTCVModel
from umei.utils import MyWandbLogger, UMeIParser

task_name = 'btcv'

def main():
    torch.set_float32_matmul_precision('high')

    parser = UMeIParser((BTCVArgs, ), use_conf=True)
    args: BTCVArgs = parser.parse_args_into_dataclasses()[0]

    if args.pretrain_path is None:
        ft_suffix = 'scratch'
    else:
        # pre-train path: .../pt type/pt param/pt datasets/run seed/last.ckpt
        # well, this is vulnerable
        ft_suffix = '/'.join(args.pretrain_path.parts[-5:-2])
    ft_suffix += f'/{args.crop}-s{args.num_seg_heads}'
    if args.spline_seg:
        ft_suffix += '-sps'
    ft_suffix += f'-{int(args.num_train_epochs)}ep-{int(args.warmup_epochs)}wu'
    ft_suffix += f'/data{args.data_ratio}'
    args.output_dir /= ft_suffix

    print(args)
    pl.seed_everything(args.seed)
    datamodule = BTCVDataModule(args)
    output_dir = args.output_dir / f'run-{args.seed}'
    output_dir.mkdir(exist_ok=True, parents=True)
    log_dir = output_dir
    if args.do_eval:
        log_dir /= f'eval-sw{args.sw_overlap}-{args.sw_blend_mode}'
    log_dir.mkdir(exist_ok=True, parents=True)
    print('real output dir:', output_dir)
    print('log dir:', log_dir)
    trainer = pl.Trainer(
        logger=MyWandbLogger(
            project=f'{task_name}-eval' if args.do_eval else task_name,
            name=str(output_dir.relative_to(args.output_root)),
            save_dir=str(log_dir),
            group=args.exp_name,
            offline=args.log_offline,
            resume=args.resume_log,
        ),
        callbacks=[
            ModelCheckpoint(
                dirpath=output_dir,
                filename=f'ep{{epoch}}-{args.monitor.replace("/", " ")}={{{args.monitor}:.3f}}',
                auto_insert_metric_name=False,
                monitor=args.monitor,
                mode=args.monitor_mode,
                verbose=True,
                save_last=True,
                save_on_train_epoch_end=False,
            ),
            ModelCheckpoint(
                dirpath=output_dir,
                filename=f'ep{{epoch}}-{args.monitor.replace("/", " ")}={{{args.monitor}:.3f}}',
                auto_insert_metric_name=False,
                verbose=True,
                save_on_train_epoch_end=False,
                save_top_k=-1,
                every_n_epochs=200,
            ),
            LearningRateMonitor(logging_interval='epoch'),
            ModelSummary(max_depth=3),
        ],
        num_nodes=args.num_nodes,
        accelerator='gpu',
        devices=torch.cuda.device_count(),
        precision=args.precision,
        benchmark=True,
        max_epochs=int(args.num_train_epochs),
        num_sanity_val_steps=args.num_sanity_val_steps,
        log_every_n_steps=5,
        check_val_every_n_epoch=args.eval_epochs,
        strategy=DDPStrategy(find_unused_parameters=args.ddp_find_unused_parameters),
        # limit_train_batches=0.1,
        # limit_val_batches=0.2,
    )
    model = BTCVModel(args)
    last_ckpt_path = args.ckpt_path
    if last_ckpt_path is None:
        last_ckpt_path = output_dir / 'last.ckpt'
        if not last_ckpt_path.exists():
            last_ckpt_path = None
    if args.do_train:
        if trainer.is_global_zero:
            conf_save_path = output_dir / 'conf.yml'
            if conf_save_path.exists():
                conf_save_path.rename(output_dir / 'conf-save.yml')
            UMeIParser.save_args_as_conf(args, conf_save_path)
        trainer.fit(model, datamodule=datamodule, ckpt_path=last_ckpt_path)
    if args.do_eval:
        trainer.test(model, ckpt_path=last_ckpt_path, datamodule=datamodule)
        # print(results)
        with open(log_dir / 'results.txt', 'w') as f:
            print('\t'.join(model.results.keys()), file=f)
            print('\t'.join(map(str, model.results.values())), file=f)
            # for k in ['test/dice-post/avg', 'test/sd-post/avg', 'test/hd95-post/avg']:
            #     print(results[0][k], file=f, end='\n' if k == 'test/hd95-post/avg' else '\t')
        with open(log_dir / 'case-results.txt', 'w') as f:
            print(*model.case_results, sep='\n', file=f)

    wandb.finish()

if __name__ == '__main__':
    main()
