from copy import deepcopy
from pathlib import Path

import omegaconf
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
import torch
import wandb

from umei.omega import parse_exp_conf
from umei.datasets.btcv.omega import BTCVExpConf
from umei.datasets.btcv.datamodule_omega import BTCVDataModule
from umei.datasets.btcv.model_omega import BTCVModel

task_name = 'btcv'

def get_exp_suffix(conf: BTCVExpConf, fold_id: int) -> str:
    suffix = Path()

    def append(name_suffix: str):
        nonlocal suffix
        suffix = suffix.with_name(f'{suffix.name}{name_suffix}')

    if conf.backbone.ckpt_path is None:
        suffix /= 'scratch'
    else:
        suffix /= '/'.join(conf.backbone.ckpt_path.parts[-3:])
    # suffix /= f's{conf.num_seg_heads}'
    # if conf.spline_seg:
    #     append('-sps')
    # append(f'-{int(conf.num_train_epochs)}ep-{int(conf.warmup_epochs)}wu')
    suffix /= f'data{conf.data_ratio}'
    suffix /= f'run-{conf.seed}/fold-{fold_id}'
    return str(suffix)

def main():
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    origin_conf = parse_exp_conf(BTCVExpConf)
    torch.set_float32_matmul_precision(origin_conf.float32_matmul_precision)

    for fold_id in origin_conf.fold_ids:
        conf = deepcopy(origin_conf)
        pl.seed_everything(conf.seed)

        # handle output_dir & log_dir
        conf.output_dir /= get_exp_suffix(conf, fold_id)
        last_ckpt_path = conf.ckpt_path
        if last_ckpt_path is None:
            last_ckpt_path = conf.output_dir / 'last.ckpt'
            if not last_ckpt_path.exists():
                last_ckpt_path = None
        if OmegaConf.is_missing(conf, 'log_dir'):
            conf.log_dir = conf.output_dir
            if conf.do_eval:
                if last_ckpt_path:
                    epoch = torch.load(last_ckpt_path)['epoch']
                else:
                    epoch = '-1'
                conf.log_dir /= f'eval-sw{conf.sw_overlap}-{conf.sw_blend_mode}{"-tta" if conf.do_tta else ""}/{epoch}'

        conf.output_dir.mkdir(exist_ok=True, parents=True)
        conf.log_dir.mkdir(exist_ok=True, parents=True)
        print('real output dir:', conf.output_dir)
        print('log dir:', conf.log_dir)

        # save config as file
        conf_save_path = conf.log_dir / 'conf.yaml'
        if conf_save_path.exists():
            conf_save_path.rename(conf_save_path.with_stem('conf-last'))
        OmegaConf.save(conf, conf_save_path)
        datamodule = BTCVDataModule(conf, fold_id)
        trainer = pl.Trainer(
            logger=WandbLogger(
                project=f'{task_name}-eval' if conf.do_eval else task_name,
                name=str(conf.output_dir.relative_to(conf.output_root)),
                save_dir=str(conf.log_dir),
                group=conf.exp_name,
                offline=conf.log_offline,
                resume=conf.resume_log,
            ),
            callbacks=[
                # save best & last
                ModelCheckpoint(
                    dirpath=conf.output_dir,
                    filename=f'best_ep{{epoch}}_{{{conf.monitor}:.3f}}',
                    auto_insert_metric_name=False,
                    monitor=conf.monitor,
                    mode=conf.monitor_mode,
                    verbose=True,
                    save_last=True,
                    save_on_train_epoch_end=False,
                ),
                ModelCheckpoint(
                    dirpath=conf.output_dir,
                    filename=f'ep{{epoch}}_{{{conf.monitor}:.3f}}',
                    auto_insert_metric_name=False,
                    verbose=True,
                    save_on_train_epoch_end=False,
                    save_top_k=-1,
                    every_n_epochs=conf.save_every_n_epochs,
                ),
                LearningRateMonitor(logging_interval='epoch'),
                ModelSummary(max_depth=3),
            ],
            gradient_clip_val=conf.gradient_clip_val,
            gradient_clip_algorithm=conf.gradient_clip_algorithm,
            num_nodes=conf.num_nodes,
            accelerator='gpu',
            devices=(n_gpu := torch.cuda.device_count()),
            precision=conf.precision,
            benchmark=True,
            max_epochs=int(conf.num_train_epochs),
            num_sanity_val_steps=conf.num_sanity_val_steps,
            strategy=DDPStrategy(find_unused_parameters=conf.ddp_find_unused_parameters) if n_gpu > 1 or conf.num_nodes > 1
            else None,
            # limit_train_batches=0.1,
            # limit_val_batches=0.2,
        )
        model = BTCVModel(conf)

        if conf.do_train:
            trainer.fit(model, datamodule=datamodule, ckpt_path=last_ckpt_path)
        if conf.do_eval:
            trainer.test(model, ckpt_path=last_ckpt_path, datamodule=datamodule)

        wandb.finish()

if __name__ == '__main__':
    main()
