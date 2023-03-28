from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
import torch
import wandb

from umei.conf import parse_exp_conf
from umei.snim.datamodule_omega import SnimDataModule
from umei.snim.model_omega import SnimModel
from umei.snim.omega import SnimConf

def main():
    conf = parse_exp_conf(SnimConf)
    torch.set_float32_matmul_precision(conf.float32_matmul_precision)
    exp_suffix = f'run-{conf.seed}'
    conf.output_dir /= exp_suffix
    conf.log_dir = conf.output_dir

    conf: SnimConf = OmegaConf.to_object(conf)
    pl.seed_everything(conf.seed)
    datamodule = SnimDataModule(conf)
    conf.output_dir.mkdir(exist_ok=True, parents=True)
    trainer = pl.Trainer(
        logger=WandbLogger(
            project='snim',
            name=f'{conf.exp_name}/{exp_suffix}',
            save_dir=conf.output_dir,
            group=conf.exp_name,
            offline=conf.log_offline,
            resume=conf.resume_log,
        ),
        callbacks=[
            ModelCheckpoint(
                dirpath=conf.output_dir,
                verbose=True,
                save_last=True,
                save_top_k=-1,
                every_n_epochs=conf.save_every_n_epochs,
                save_on_train_epoch_end=True,
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
        log_every_n_steps=conf.log_every_n_steps,
        strategy=DDPStrategy(find_unused_parameters=conf.ddp_find_unused_parameters) if n_gpu > 1 or conf.num_nodes > 1
        else None,
        num_sanity_val_steps=conf.num_sanity_val_steps,
    )
    model = SnimModel(conf)
    last_ckpt_path = conf.ckpt_path
    if last_ckpt_path is None:
        last_ckpt_path = conf.output_dir / 'last.ckpt'
        if not last_ckpt_path.exists():
            last_ckpt_path = None
    if conf.do_train:
        conf_save_path = conf.output_dir / 'conf.yaml'
        if trainer.is_global_zero:
            if conf_save_path.exists():
                conf_save_path.rename(conf_save_path.with_stem('conf-last'))
        OmegaConf.save(conf, conf_save_path)
        trainer.fit(model, datamodule=datamodule, ckpt_path=last_ckpt_path)

    wandb.finish()

if __name__ == '__main__':
    main()
