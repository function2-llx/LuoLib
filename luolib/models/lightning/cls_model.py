from collections.abc import Mapping, Hashable
from typing import TypeAlias

import torch
import torchmetrics
from torch import nn
from torchmetrics.utilities.enums import AverageMethod

from luolib.conf import ClsExpConf
from luolib.utils import DataSplit, DataKey
from .model_base import ExpModelBase

MetricsCollection: TypeAlias = Mapping[str, torchmetrics.Metric]

# multi-class classification model base
class ClsModel(ExpModelBase):
    conf: ClsExpConf
    micro_metrics: MetricsCollection | None = None

    @property
    def val_splits(self):
        return [DataSplit.VAL, DataSplit.TEST]

    def __init__(self, conf: ClsExpConf):
        super().__init__(conf)
        self.cls_loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(conf.cls_weights))
        self.cls_metrics: Mapping[DataSplit, MetricsCollection] = nn.ModuleDict({
            split: self.create_metrics(conf.num_cls_classes)
            for split in self.val_splits
        })

    @property
    def flip_keys(self) -> list[Hashable]:
        raise NotImplementedError

    def cal_logit_impl(self, batch: dict):
        raise NotImplementedError

    def cal_logit(self, batch):
        conf = self.conf
        logit = self.cal_logit_impl(batch)
        if self.trainer.testing and conf.do_tta:
            for flip_idx in self.tta_flips:
                batch = dict(batch)
                for key in self.flip_keys:
                    batch[key] = torch.flip(batch[key], flip_idx)
                logit += self.cal_logit_impl(batch)
            logit /= len(self.tta_flips + 1)
        return logit

    def training_step(self, batch, _batch_idx: int):
        logit = self.cal_logit(batch)
        return self.cls_loss_fn(logit, batch[DataKey.CLS])

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        for split in self.val_splits:
            self.reset_metrics(split)

    def validation_step(self, batch: dict, _batch_idx: int, dataloader_idx: int) -> torch.Tensor:
        logit = self.cal_logit(batch)
        label = batch[DataKey.CLS]
        loss = self.cls_loss_fn(logit, label)
        split = self.val_splits[dataloader_idx]
        self.log(f'{split}/loss', loss)
        prob = logit.softmax(dim=-1)
        self.accumulate_metrics(prob, label, self.cls_metrics[split])
        return prob

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        for split in self.val_splits:
            self.log_metrics(split)

    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        self.reset_metrics(DataSplit.TEST)

    def test_step(self, batch: dict, _batch_idx: int, _dl_idx: int):
        prob = self.cal_logit(batch).softmax(dim=-1)
        label = batch[DataKey.CLS]
        self.accumulate_metrics(prob, label, self.cls_metrics[DataSplit.TEST])
        if self.micro_metrics is not None:
            self.accumulate_metrics(prob, label, self.micro_metrics)

    @staticmethod
    def accumulate_metrics(prob: torch.Tensor, label: torch.Tensor, metrics: MetricsCollection):
        for k, metric in metrics.items():
            metric(prob, label)

    def reset_metrics(self, split: DataSplit):
        for metric in self.cls_metrics[split].values():
            metric.reset()

    @property
    def cls_names(self):
        return list(range(self.conf.num_cls_classes))

    def log_metrics(self, split: DataSplit):
        for k, metric in self.cls_metrics[split].items():
            m = metric.compute()
            if metric.average == AverageMethod.NONE:
                for i, cls in enumerate(self.cls_names):
                    self.log(f'{split}/{k}/{cls}', m[i], sync_dist=True)
                self.log(f'{split}/{k}/avg', m.mean(), sync_dist=True)
            else:
                self.log(f'{split}/{k}', m, sync_dist=True)

    @staticmethod
    def create_metrics(num_classes: int) -> Mapping[str, torchmetrics.Metric]:
        return nn.ModuleDict({
            k: metric_cls(task='multiclass', num_classes=num_classes, average=average)
            for k, metric_cls, average in [
                ('auroc', torchmetrics.AUROC, AverageMethod.NONE),
                ('recall', torchmetrics.Recall, AverageMethod.NONE),
                ('precision', torchmetrics.Precision, AverageMethod.NONE),
                ('f1', torchmetrics.F1Score, AverageMethod.NONE),
                ('acc', torchmetrics.Accuracy, AverageMethod.MICRO),
            ]
        })

    @staticmethod
    def metrics_report(metrics: Mapping[str, torchmetrics.Metric], names: list[str]):
        report = {}
        for k, metric in metrics.items():
            m = metric.compute()
            if metric.average == AverageMethod.NONE:
                for i, cls in enumerate(names):
                    report[f'{k}/{cls}'] = m[i].item()
                report[f'{k}/avg'] = m.mean().item()
            else:
                report[k] = m.item()

        return report
