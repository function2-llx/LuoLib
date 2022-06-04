import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
import torch.cuda
import wandb

from umei.datasets.amos import AmosArgs, AmosDataModule
from umei.datasets.amos.model import AmosModel
from umei.utils import MyWandbLogger, UMeIParser

def main():
    parser = UMeIParser((AmosArgs, ), use_conf=True)
    args: AmosArgs = parser.parse_args_into_dataclasses()[0]
    datamodule = AmosDataModule(args)
    for val_fold_id in range(datamodule.num_cv_folds):
        pl.seed_everything(args.seed)
        datamodule.val_id = val_fold_id
        output_dir = args.output_dir / f'fold{val_fold_id}'
        output_dir.mkdir(exist_ok=True, parents=True)
        conf_save_path = output_dir / 'conf.yml'
        # save tmp to handle multiple processes
        conf_tmp_path = output_dir / 'conf-tmp.yml'
        if conf_save_path.exists() and not args.overwrite_output_dir:
            print(f'{conf_save_path} exists (fit complete), skip')
            continue

        model = AmosModel(args)
        if (last_ckpt_path := output_dir / 'last.ckpt').exists():
            model.load_from_checkpoint(str(last_ckpt_path), args=args, encoder=model.encoder, decoder=model.decoder)
            # latest_run_id = (output_dir / 'wandb/latest-run').resolve().name.split('-')[-1]
            # print('latest wandb id:', latest_run_id)
        else:
            last_ckpt_path = None
            # latest_run_id = None
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
            # num_nodes=args.num_nodes,
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
        UMeIParser.save_args_as_conf(args, conf_tmp_path)
        trainer.fit(model, datamodule=datamodule, ckpt_path=last_ckpt_path)
        if conf_tmp_path.exists():
            conf_tmp_path.rename(conf_save_path)

        wandb.finish()

if __name__ == '__main__':
    main()
