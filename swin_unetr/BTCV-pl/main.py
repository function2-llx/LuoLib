# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from functools import partial
import os
from pathlib import Path

import numpy as np
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed
import pytorch_lightning as pl

from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import AsDiscrete
from monai.utils.enums import MetricReduction

from trainer import AmosModel
from umei.utils import MyWandbLogger
from utils.data_utils import get_loader

parser = argparse.ArgumentParser(description='Swin UNETR segmentation pipeline')
parser.add_argument('--checkpoint', default=None, help='start training from saved checkpoint')
parser.add_argument('--logdir', default='test', type=str, help='directory to save the tensorboard logs')
parser.add_argument('--pretrained_dir', default='./pretrained_models/', type=str, help='pretrained checkpoint directory')
parser.add_argument('--data_dir', default='/dataset/dataset0/', type=str, help='dataset directory')
parser.add_argument('--json_list', default='dataset_0.json', type=str, help='dataset json file')
parser.add_argument('--pretrained_model_name', default='swin_unetr.epoch.b4_5000ep_f48_lr2e-4_pretrained.pt', type=str, help='pretrained model name')
parser.add_argument('--save_checkpoint', action='store_true', help='save checkpoint during training')
parser.add_argument('--max_epochs', default=5000, type=int, help='max number of training epochs')
parser.add_argument('--batch_size', default=1, type=int, help='number of batch size')
parser.add_argument('--sw_batch_size', default=4, type=int, help='number of sliding window batch size')
parser.add_argument('--optim_lr', default=1e-4, type=float, help='optimization learning rate')
parser.add_argument('--optim_name', default='adamw', type=str, help='optimization algorithm')
parser.add_argument('--reg_weight', default=1e-5, type=float, help='regularization weight')
parser.add_argument('--momentum', default=0.99, type=float, help='momentum')
parser.add_argument('--noamp', action='store_true', help='do NOT use amp for training')
parser.add_argument('--val_every', default=100, type=int, help='validation frequency')
parser.add_argument('--world_size', default=1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str, help='distributed url')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--workers', default=8, type=int, help='number of workers')
parser.add_argument('--feature_size', default=48, type=int, help='feature size')
parser.add_argument('--in_channels', default=1, type=int, help='number of input channels')
parser.add_argument('--out_channels', default=14, type=int, help='number of output channels')
parser.add_argument('--use_normal_dataset', action='store_true', help='use monai Dataset class')
parser.add_argument('--a_min', default=-175.0, type=float, help='a_min in ScaleIntensityRanged')
parser.add_argument('--a_max', default=250.0, type=float, help='a_max in ScaleIntensityRanged')
parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
parser.add_argument('--space_x', default=1.5, type=float, help='spacing in x direction')
parser.add_argument('--space_y', default=1.5, type=float, help='spacing in y direction')
parser.add_argument('--space_z', default=2.0, type=float, help='spacing in z direction')
parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')
parser.add_argument('--dropout_rate', default=0.0, type=float, help='dropout rate')
parser.add_argument('--dropout_path_rate', default=0.0, type=float, help='drop path rate')
parser.add_argument('--RandFlipd_prob', default=0.2, type=float, help='RandFlipd aug probability')
parser.add_argument('--RandRotate90d_prob', default=0.2, type=float, help='RandRotate90d aug probability')
parser.add_argument('--RandScaleIntensityd_prob', default=0.1, type=float, help='RandScaleIntensityd aug probability')
parser.add_argument('--RandShiftIntensityd_prob', default=0.1, type=float, help='RandShiftIntensityd aug probability')
parser.add_argument('--infer_overlap', default=0.25, type=float, help='sliding window inference overlap')
parser.add_argument('--lrschedule', default='warmup_cosine', type=str, help='type of learning rate scheduler')
parser.add_argument('--warmup_epochs', default=50, type=int, help='number of warmup epochs')
parser.add_argument('--resume_ckpt', action='store_true', help='resume training from pretrained checkpoint')
parser.add_argument('--smooth_dr', default=1e-6, type=float, help='constant added to dice denominator to avoid nan')
parser.add_argument('--smooth_nr', default=0.0, type=float, help='constant added to dice numerator to avoid zero')
parser.add_argument('--use_checkpoint', action='store_true', help='use gradient checkpointing to save memory')
parser.add_argument('--use_ssl_pretrained', action='store_true', help='use self-supervised pretrained weights')
parser.add_argument('--spatial_dims', default=3, type=int, help='spatial dimension of input data')
parser.add_argument('--squared_dice', action='store_true', help='use squared Dice')

parser.add_argument('--exp_name', type=str, default=None)
parser.add_argument('--log_offline', action='store_true')
parser.add_argument('--split_model', action='store_true')

def main():
    args = parser.parse_args()
    if args.exp_name is None:
        args.exp_name = args.logdir
    args.amp = not args.noamp
    args.logdir = './runs/' + args.logdir
    Path(args.logdir).mkdir(parents=True, exist_ok=True)
    main_worker(args=args)

def main_worker(args):
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, suppress=True)

    torch.backends.cudnn.benchmark = True
    args.test_mode = False

    print('Batch size is:', args.batch_size, 'epochs', args.max_epochs)

    amos_model = AmosModel(args=args)
    trainer = pl.Trainer(
        logger=MyWandbLogger(
            project='amos',
            name=args.exp_name,
            save_dir=args.logdir,
            group='monai',
            offline=args.log_offline,
            resume=True,
        ),
        callbacks=[
            ModelCheckpoint(
                dirpath=args.logdir,
                filename='val-acc={val/acc:.3f}',
                auto_insert_metric_name=False,
                monitor='val/acc',
                mode='max',
                verbose=True,
                save_last=True,
                save_on_train_epoch_end=False,
            ),
            LearningRateMonitor(logging_interval='epoch')
        ],
        gpus=1,
        precision=16,
        benchmark=True,
        max_epochs=args.max_epochs,
        num_sanity_val_steps=5,
        log_every_n_steps=5,
        check_val_every_n_epoch=args.val_every,
    )
    trainer.fit(amos_model)

if __name__ == '__main__':
    main()
