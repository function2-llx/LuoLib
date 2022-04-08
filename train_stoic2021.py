from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
import torch
from torch import nn
from torch.nn import functional as F
import torchmetrics
import wandb

from umei import UEncoderBase, UMeI
from umei.datasets import Stoic2021DataModule
from umei.utils import UMeIArgs, UMeIParser

torch.multiprocessing.set_sharing_strategy('file_system')

def build_encoder(args: UMeIArgs) -> UEncoderBase:
    from monai.networks import nets
    if args.model_name == 'resnet':
        resnet = getattr(nets, f'resnet{args.model_depth}')
        model: nn.Module = resnet(
            n_input_channels=args.num_input_channels,
            feed_forward=False,
            shortcut_type=args.resnet_shortcut,
        )
        if args.pretrain_path is not None:
            # assume pre-trained weights are from https://github.com/Tencent/MedicalNet
            dp_model = nn.DataParallel(model)
            state_dict = dp_model.state_dict()
            pretrain_state_dict = torch.load(args.pretrain_path, map_location='cpu')['state_dict']
            state_dict.update({
                k: v for k, v in pretrain_state_dict.items()
                if k in state_dict and k != 'module.conv1.weight'
            })
            dp_model.load_state_dict(state_dict)
            model: nets.ResNet = dp_model.module  # type: ignore

            # handle number of input channels that is possible different from the pre-trained model
            for attr in ['weight', 'bias']:
                param: Optional[nn.Parameter] = getattr(model.conv1, attr, None)
                pretrain_param_data: Optional[torch.Tensor] = getattr(pretrain_state_dict, f'module.conv1.{attr}', None)
                if param is not None and pretrain_param_data is not None:
                    param.data = pretrain_param_data.repeat(1, args.num_input_channels)
        return model
    else:
        raise NotImplementedError

# cannot wait: https://github.com/PyTorchLightning/pytorch-lightning/pull/12172/
class MyWandbLogger(WandbLogger):
    @WandbLogger.name.getter
    def name(self) -> Optional[str]:
        return self._experiment.name if self._experiment else self._name

class Stoic2021(UMeI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.severity_auc = torchmetrics.AUROC(pos_label=1)
        self.positivity_auc = torchmetrics.AUROC(pos_label=1)

    def validation_step(self, batch: dict[str, torch.Tensor], *args, **kwargs):
        output = super().validation_step(batch, *args, **kwargs)
        pred = F.softmax(output['cls_logit'], dim=1)
        positive_idx: torch.Tensor = batch[self.args.cls_key] >= 1  # type: ignore
        if positive_idx.sum() > 0:
            severity_pred = pred[positive_idx, 2] / pred[positive_idx, 1:].sum(dim=1)
            self.severity_auc(
                preds=severity_pred,
                target=batch[self.args.cls_key][positive_idx] == 2,
            )
            self.log('val/auc-severity', self.severity_auc)
        self.positivity_auc(
            preds=pred[:, 1:].sum(dim=1),
            target=batch[self.args.cls_key] >= 1,
        )
        self.log('val/auc-positivity', self.positivity_auc)
        return output

def main():
    parser = UMeIParser((UMeIArgs,), use_conf=True)
    args: UMeIArgs = parser.parse_args_into_dataclasses()[0]
    datamodule = Stoic2021DataModule(args)
    for fold_id in range(args.num_folds):
        output_dir = args.output_dir / f'fold{fold_id}'
        output_dir.mkdir(exist_ok=True, parents=True)
        datamodule.val_id = fold_id

        model = Stoic2021(args, encoder=build_encoder(args))
        trainer = pl.Trainer(
            logger=MyWandbLogger(
                name=f'{args.exp_name}/fold{fold_id}',
                save_dir=str(output_dir),
            ) if args.log else None,
            gpus=args.n_gpu,
            precision=args.precision,
            benchmark=True,
            max_epochs=int(args.num_train_epochs),
            callbacks=[
                ModelCheckpoint(
                    dirpath=output_dir,
                    filename=f'{args.monitor}={{val/{args.monitor}:.3f}}',
                    auto_insert_metric_name=False,
                    monitor=f'val/{args.monitor}',
                    verbose=True,
                    save_last=True,
                    save_top_k=2,
                ),
            ],
            num_sanity_val_steps=0,
            log_every_n_steps=20,
            strategy=DDPStrategy(find_unused_parameters=False),
            # limit_train_batches=0.1,
            # limit_val_batches=0.2,
        )
        trainer.fit(model, datamodule=datamodule)

        wandb.finish()

if __name__ == '__main__':
    main()
