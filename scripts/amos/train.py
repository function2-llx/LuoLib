import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
import torch
import wandb

from umei.datasets.amos import AmosArgs, AmosDataModule, AmosModel
from umei.utils import MyWandbLogger, UMeIParser

def main():
    parser = UMeIParser((AmosArgs, ), use_conf=True)
    args: AmosArgs = parser.parse_args_into_dataclasses()[0]
    print(args)
    datamodule = AmosDataModule(args)
    for val_fold_id in range(datamodule.num_cv_folds):
        if val_fold_id not in args.fold_ids:
            continue
        pl.seed_everything(args.seed)
        datamodule.val_id = val_fold_id
        output_dir = args.output_dir / f'run-{args.seed}' / f'fold{val_fold_id}'
        output_dir.mkdir(exist_ok=True, parents=True)
        trainer = pl.Trainer(
            logger=MyWandbLogger(
                project='amos',
                name=f'{args.exp_name}/runs-{args.seed}/fold{val_fold_id}',
                save_dir=str(output_dir),
                group=args.exp_name,
                offline=args.log_offline,
                resume=True,
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
            num_sanity_val_steps=args.num_sanity_val_steps,
            log_every_n_steps=5,
            check_val_every_n_epoch=args.eval_epochs,
            strategy=DDPStrategy(find_unused_parameters=args.ddp_find_unused_parameters),
            # limit_train_batches=0.1,
            # limit_val_batches=0.2,
        )
        model = AmosModel(args)
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
            if args.use_monai:
                model.seg_heads[0].load_state_dict({
                    k[4:]: v
                    for k, v in torch.load(args.decoder_pretrain_path)['state_dict'].items()
                    if k.startswith('out.')
                })
                print('DEBUG: load pre-trained seg head')
            trainer.test(model, ckpt_path=last_ckpt_path, datamodule=datamodule)

        wandb.finish()

if __name__ == '__main__':
    main()
