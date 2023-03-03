from pathlib import Path

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

def get_exp_suffix(conf: BTCVExpConf) -> str:
    suffix = Path()

    def append(name_suffix: str):
        nonlocal suffix
        suffix = suffix.with_name(f'{suffix.name}{name_suffix}')

    if conf.backbone.ckpt_path is None:
        suffix /= 'scratch'
    else:
        raise NotImplementedError
    suffix /= f's{conf.num_seg_heads}'
    if conf.spline_seg:
        append('-sps')
    # append(f'-{int(conf.num_train_epochs)}ep-{int(conf.warmup_epochs)}wu')
    suffix /= f'data{conf.data_ratio}'
    suffix /= f'run-{conf.seed}'
    return str(suffix)

def main():
    conf = parse_exp_conf(BTCVExpConf)
    torch.set_float32_matmul_precision(conf.float32_matmul_precision)
    pl.seed_everything(conf.seed)

    # handle output_dir & log_dir
    conf.output_dir /= get_exp_suffix(conf)
    if OmegaConf.is_missing(conf, 'log_dir'):
        conf.log_dir = conf.output_dir
        if conf.do_eval:
            conf.log_dir /= f'eval-sw{conf.sw_overlap}-{conf.sw_blend_mode}{"-tta" if conf.do_tta else ""}'
    conf.output_dir.mkdir(exist_ok=True, parents=True)
    conf.log_dir.mkdir(exist_ok=True, parents=True)
    print('real output dir:', conf.output_dir)
    print('log dir:', conf.log_dir)

    # save config as file
    conf_save_path = conf.output_dir / 'conf.yml'
    if conf_save_path.exists():
        conf_save_path.rename(conf_save_path.with_stem('conf-last'))
    OmegaConf.save(conf, conf_save_path)
    datamodule = BTCVDataModule(conf)
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
            ModelCheckpoint(
                dirpath=conf.output_dir,
                filename=f'ep{{epoch}}-{conf.monitor.replace("/", " ")}={{{conf.monitor}:.3f}}',
                auto_insert_metric_name=False,
                monitor=conf.monitor,
                mode=conf.monitor_mode,
                verbose=True,
                save_last=True,
                save_on_train_epoch_end=False,
            ),
            ModelCheckpoint(
                dirpath=conf.output_dir,
                filename=f'ep{{epoch}}-{conf.monitor.replace("/", " ")}={{{conf.monitor}:.3f}}',
                auto_insert_metric_name=False,
                verbose=True,
                save_on_train_epoch_end=False,
                save_top_k=-1,
                every_n_epochs=20,
            ),
            LearningRateMonitor(logging_interval='epoch'),
            ModelSummary(max_depth=3),
        ],
        num_nodes=conf.num_nodes,
        accelerator='gpu',
        devices=torch.cuda.device_count(),
        precision=conf.precision,
        benchmark=True,
        max_epochs=int(conf.num_train_epochs),
        num_sanity_val_steps=conf.num_sanity_val_steps,
        strategy=DDPStrategy(find_unused_parameters=conf.ddp_find_unused_parameters),
        # limit_train_batches=0.1,
        # limit_val_batches=0.2,
    )
    model = BTCVModel(conf)
    last_ckpt_path = conf.ckpt_path
    if last_ckpt_path is None:
        last_ckpt_path = conf.output_dir / 'last.ckpt'
        if not last_ckpt_path.exists():
            last_ckpt_path = None
    if conf.do_train:
        trainer.fit(model, datamodule=datamodule, ckpt_path=last_ckpt_path)
    if conf.do_eval:
        trainer.test(model, ckpt_path=last_ckpt_path, datamodule=datamodule)

    wandb.finish()

if __name__ == '__main__':
    main()
