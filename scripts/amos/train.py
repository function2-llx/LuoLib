from pytorch_lightning.strategies import DDPStrategy
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch.cuda
import wandb

from umei.datasets.amos import AmosDataModule, AmosArgs
from umei.datasets.amos.model import AmosModel
from umei.model import build_decoder, build_encoder
from umei.utils import MyWandbLogger, UMeIParser

def main():
    parser = UMeIParser((AmosArgs, ), use_conf=True)
    args: AmosArgs = parser.parse_args_into_dataclasses()[0]
    datamodule = AmosDataModule(args)
    for val_fold_id in range(datamodule.num_cv_folds):
        pl.seed_everything(args.seed)
        output_dir = args.output_dir / f'fold{val_fold_id}'
        output_dir.mkdir(exist_ok=True, parents=True)
        conf_save_path = output_dir / 'conf.yml'
        # save tmp to handle multiple processes
        conf_tmp_path = output_dir / 'conf-tmp.yml'
        if conf_save_path.exists() and not args.overwrite_output_dir:
            print(f'{conf_save_path} exists (fit complete), skip')
            continue
        datamodule.val_id = val_fold_id
        encoder = build_encoder(args)
        decoder = build_decoder(args)
        model = AmosModel(args, encoder, decoder)
        if (last_ckpt_path := output_dir / 'last.ckpt').exists():
            model.load_from_checkpoint(str(last_ckpt_path), args=args, encoder=encoder, decoder=decoder)
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
                    filename=f'{args.monitor}={{{args.monitor}:.3f}}',
                    auto_insert_metric_name=False,
                    monitor=args.monitor,
                    mode=args.monitor_mode,
                    verbose=True,
                    save_last=True,
                    save_on_train_epoch_end=False,
                ),
                EarlyStopping(
                    monitor=args.monitor,
                    patience=3 * args.patience,
                    mode=args.monitor_mode,
                    verbose=True,
                    check_on_train_epoch_end=False,
                ),
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
