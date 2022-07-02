from argparse import Namespace
from collections.abc import Callable
from typing import Optional

from pytorch_lightning import LightningModule
from torch import nn
from torch.optim import Optimizer

from monai.data import DataLoader, decollate_batch
from monai.metrics import CumulativeIterationMetric
from utils.utils import AverageMeter

class AmosModel(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        loss_func: nn.Module,
        acc_func: CumulativeIterationMetric,
        args: Namespace,
        model_inferer: Optional[Callable] = None,
        scheduler=None,
        post_label=None,
        post_pred=None,
    ):
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.acc_func = acc_func
        self.args = args
        self.model_inferer = model_inferer
        self.scheduler = scheduler
        self.post_label = post_label
        self.post_pred = post_pred

        self.run_loss = AverageMeter()
        self.run_acc = AverageMeter()

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def configure_optimizers(self):
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': self.scheduler
        }

    def on_train_epoch_start(self):
        self.run_loss.reset()

    def training_step(self, batch_data, idx, *args, **kwargs):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data['image'], batch_data['label']
        for param in self.model.parameters():
            param.grad = None
        logits = self.model(data)
        loss = self.loss_func(logits, target)
        self.log('train/loss', loss.item())
        self.run_loss.update(loss.item(), n=data.shape[0])
        return loss

    def on_train_epoch_end(self):
        # self.log('train/loss', self.run_loss.avg)
        for param in self.model.parameters():
            param.grad = None

    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer: Optimizer, optimizer_idx: int):
        pass

    def on_validation_epoch_start(self):
        self.run_acc.reset()

    def validation_step(self, batch_data, idx, *args, **kwargs):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data['image'], batch_data['label']

        if self.model_inferer is not None:
            logits = self.model_inferer(data)
        else:
            logits = self.model(data)
        val_labels_list = decollate_batch(target)
        val_labels_convert = [self.post_label(val_label_tensor) for val_label_tensor in val_labels_list]
        val_outputs_list = decollate_batch(logits)
        val_output_convert = [self.post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
        self.acc_func.reset()
        self.acc_func(y_pred=val_output_convert, y=val_labels_convert)
        acc, not_nans = self.acc_func.aggregate()

        self.run_acc.update(acc.item(), n=not_nans.item())

    def validation_epoch_end(self, _outputs):
        self.log('val/acc', self.run_acc.avg * 100)
