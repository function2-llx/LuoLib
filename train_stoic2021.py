from numpy.random import SeedSequence
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy
import torch
import wandb

from umei.datasets import Stoic2021DataModule, Stoic2021Args, Stoic2021Model
from umei.model import build_encoder
from umei.utils import MyWandbLogger, UMeIParser

torch.multiprocessing.set_sharing_strategy('file_system')

def main():
    parser = UMeIParser((Stoic2021Args,), use_conf=True)
    args: Stoic2021Args = parser.parse_args_into_dataclasses()[0]
    datamodule = Stoic2021DataModule(args)

    for run, seed in zip(
        range(args.num_runs),
        SeedSequence(args.seed).generate_state(args.num_runs),
    ):
        for val_fold_id in range(datamodule.num_cv_folds):
            pl.seed_everything(seed)

            output_dir = args.output_dir / f'fold{val_fold_id}' / f'run{run}'
            output_dir.mkdir(exist_ok=True, parents=True)

            datamodule.val_id = val_fold_id

            encoder = build_encoder(args)
            model = Stoic2021Model(args, encoder=encoder)
            if (last_ckpt_path := output_dir / 'last.ckpt').exists():
                model.load_from_checkpoint(str(last_ckpt_path), args=args, encoder=encoder)
                latest_run_id = (output_dir / 'wandb/latest-run').resolve().name.split('-')[-1]
                print('latest wandb id:', latest_run_id)
            else:
                last_ckpt_path = None
                latest_run_id = None

            trainer = pl.Trainer(
                logger=MyWandbLogger(
                    name=f'{args.exp_name}/fold{val_fold_id}/run{run}',
                    id=latest_run_id,
                    save_dir=str(output_dir),
                    group=args.exp_name,
                ),
                callbacks=[
                    ModelCheckpoint(
                        dirpath=output_dir,
                        filename=f'{args.monitor}={{combined/{args.monitor}:.3f}}',
                        auto_insert_metric_name=False,
                        monitor=f'combined/{args.monitor}',
                        mode=args.monitor_mode,
                        verbose=True,
                        save_last=True,
                        save_top_k=2,
                        save_on_train_epoch_end=False,
                    ),
                    EarlyStopping(
                        monitor=f'combined/{args.monitor}',
                        patience=3 * args.patience,
                        mode=args.monitor_mode,
                        verbose=True,
                        check_on_train_epoch_end=False,
                    ),
                ],
                num_nodes=args.num_nodes,
                gpus=args.n_gpu,
                precision=args.precision,
                benchmark=True,
                max_epochs=int(args.num_train_epochs),
                num_sanity_val_steps=0,
                strategy=DDPStrategy(find_unused_parameters=False),
                # limit_train_batches=0.1,
                # limit_val_batches=0.2,
            )
            trainer.fit(model, datamodule=datamodule, ckpt_path=last_ckpt_path)

            wandb.finish()

if __name__ == '__main__':
    main()
