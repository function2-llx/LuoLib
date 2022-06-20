import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
import wandb
import torch

from umei.datasets.amos import AmosArgs, AmosDataModule, AmosModel
from umei.utils import MyWandbLogger, UMeIParser

def main():
    parser = UMeIParser((AmosArgs, ), use_conf=True)
    args: AmosArgs = parser.parse_args_into_dataclasses()[0]
    datamodule = AmosDataModule(args)
    for val_fold_id in range(datamodule.num_cv_folds):
        if val_fold_id not in args.fold_ids:
            continue
        pl.seed_everything(args.seed)
        datamodule.val_id = val_fold_id
        output_dir = args.output_dir / f'fold{val_fold_id}'
        output_dir.mkdir(exist_ok=True, parents=True)
        trainer = pl.Trainer(
            logger=MyWandbLogger(
                project='amos',
                name=f'{args.exp_name}/fold{val_fold_id}',
                save_dir=str(output_dir),
                group=args.exp_name,
            ),
            callbacks=[
                ModelCheckpoint(
                    dirpath=output_dir,
                    filename=f'{args.monitor.replace("/", " ")}={{{args.monitor}:.3f}}',
                    auto_insert_metric_name=False,
                    monitor=args.monitor,
                    mode=args.monitor_mode,
                    verbose=True,
                    save_last=True,
                    save_on_train_epoch_end=False,
                ),
                LearningRateMonitor(logging_interval='epoch')
            ],
            num_nodes=args.num_nodes,
            gpus=torch.cuda.device_count(),
            precision=args.precision,
            benchmark=True,
            max_epochs=int(args.num_train_epochs),
            num_sanity_val_steps=5,
            log_every_n_steps=5,
            strategy=DDPStrategy(find_unused_parameters=args.ddp_find_unused_parameters),
            # limit_train_batches=0.1,
            # limit_val_batches=0.2,
        )
        model = AmosModel(args)
        last_ckpt_path = output_dir / 'last.ckpt'
        if not last_ckpt_path.exists():
            last_ckpt_path = None
        # if not args.overwrite_output_dir and (last_ckpt_path := output_dir / 'last.ckpt').exists():
        #     model = AmosModel.load_from_checkpoint(str(last_ckpt_path), args=args)
        #     # latest_run_id = (output_dir / 'wandb/latest-run').resolve().name.split('-')[-1]
        #     # print('latest wandb id:', latest_run_id)
        # else:
        #     last_ckpt_path = None
        #     # latest_run_id = None
        if args.do_train:
            conf_save_path = output_dir / 'conf.yml'
            if not conf_save_path.exists() and not args.overwrite_output_dir:
                print(f'{conf_save_path} exists and overwrite not allowed, skip')
                continue
            UMeIParser.save_args_as_conf(args, conf_save_path)
            trainer.fit(model, datamodule=datamodule, ckpt_path=last_ckpt_path)
        if args.do_eval:
            trainer.validate(model, ckpt_path=last_ckpt_path, datamodule=datamodule)

        wandb.finish()

    # if args.do_eval:
    #     for val_fold_id in range(datamodule.num_cv_folds):
    #         if val_fold_id not in args.fold_ids:
    #             continue
    #         pl.seed_everything(args.seed)
    #         datamodule.val_id = val_fold_id
    #         trainer.validate(model, datamodule)


if __name__ == '__main__':
    main()
