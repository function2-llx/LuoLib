import torch
from torch import nn
from torch.nn import functional as F
import torchmetrics

from umei import UEncoderBase, UMeI
from .args import Stoic2021Args

class Stoic2021Model(UMeI):
    def __init__(self, args: Stoic2021Args, encoder: UEncoderBase):
        super().__init__(args, encoder)
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
        splits_output = super().validation_step(splits_batch)
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
        output = self.forward(batch)
        pred = F.softmax(output['cls_logit'], dim=1)
        return {
            'severity_pred': pred[:, 2] / pred[:, 1:].sum(dim=1),
            'positivity_pred': pred[:, 1:].sum(dim=1),
            self.args.cls_key: batch[self.args.cls_key],
        }
