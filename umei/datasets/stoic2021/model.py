import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import RAdam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchmetrics

from umei import UMeI
from .args import Stoic2021Args
from monai.umei import Backbone, BackboneOutput

class Stoic2021Model(UMeI):
    args: Stoic2021Args

    def __init__(self, args: Stoic2021Args):
        super().__init__(args, has_decoder=False)
        self.cls_loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1, args.cls_weight, args.cls_weight]).float())
        self.severity_auc = nn.ModuleDict({
            k: torchmetrics.AUROC(pos_label=1)
            for k in ['val', 'test', 'combined']
        })
        self.positivity_auc = nn.ModuleDict({
            k: torchmetrics.AUROC(pos_label=1)
            for k in ['val', 'test', 'combined']
        })

    def validation_step(self, splits_batch: dict[str, dict[str, torch.Tensor]], *args, **kwargs):
        # splits_output = super().validation_step(splits_batch)
        splits_output = {}
        for split, batch in splits_batch.items():
            batch_size = batch[self.args.img_key].shape[0]
            output = self.forward(batch)
            for k in ['cls_loss', 'seg_loss']:
                if k in output:
                    self.log(f'{split}/{k}', output[k], batch_size=batch_size)
                    if self.args.use_test_fold:
                        self.log(f'combined/{k}', output[k], batch_size=batch_size)
            splits_output[split] = output

        for split, batch in splits_batch.items():
            batch_size = batch[self.args.img_key].shape[0]
            output = splits_output[split]
            pred = F.softmax(output['cls_logit'], dim=1)

            positive_idx: torch.Tensor = batch[self.args.cls_key] >= 1  # type: ignore
            if positive_idx.sum() > 0:
                # P(severe | positive)
                severity_pred = pred[positive_idx, 2] / pred[positive_idx, 1:].sum(dim=1)
                severity_target = batch[self.args.cls_key][positive_idx] == 2
                for k in [split, 'combined']:
                    self.severity_auc[k](
                        preds=severity_pred,
                        target=severity_target,
                    )
                    self.log(f'{k}/auc-severity', self.severity_auc[k], batch_size=batch_size)  # type: ignore

            for k in [split, 'combined']:
                positivity_pred = pred[:, 1:].sum(dim=1)
                positivity_target = batch[self.args.cls_key] >= 1
                self.positivity_auc[k](
                    preds=positivity_pred,
                    target=positivity_target,
                )
                self.log(f'{k}/auc-positivity', self.positivity_auc[k], batch_size=batch_size)  # type: ignore

        return splits_output

    def predict_step(self, batch: dict[str, torch.Tensor], *args, **kwargs):
        encoder_out: BackboneOutput = self.encoder(batch[self.args.img_key])
        cls_logit = self.cls_head(torch.cat((encoder_out.cls_feature, batch[self.args.clinical_key]), dim=1))
        pred = F.softmax(cls_logit, dim=1)
        return {
            'severity_pred': pred[:, 2] / pred[:, 1:].sum(dim=1),
            'positivity_pred': pred[:, 1:].sum(dim=1),
        }

    def configure_optimizers(self):
        optimizer = RAdam(self.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': ReduceLROnPlateau(
                    optimizer,
                    mode=self.args.monitor_mode,
                    factor=self.args.lr_reduce_factor,
                    patience=self.args.patience,
                    verbose=True,
                ),
                'monitor': f'combined/{self.args.monitor}',
            }
        }
