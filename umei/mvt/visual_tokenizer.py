from typing import Any

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as torch_f

from umei.utils import channel_last
from umei.models.backbones.swin import SwinBackbone
from umei.models.init import init_linear_conv
from .args import MVTModelArgs
from .decoder import SwinVQDecoder

from .vector_quantizer import NormEMAVectorQuantizer

class VisualTokenizer(pl.LightningModule):
    teacher_model: nn.Module | None

    def __init__(self, conf: MVTModelArgs):
        super().__init__()
        self.conf = conf

        # encoder & decode params
        print('Final encoder config', conf.encoder)
        self.encoder = SwinBackbone(**conf.encoder)

        print('Final decoder config', conf.decoder)
        self.decoder = SwinVQDecoder(**conf.decoder)

        self.quantizer = NormEMAVectorQuantizer(
            conf.num_tokens,
            conf.embedding_dim,
            **conf.quantizer,
        )

        # Teacher model setting
        match self.conf.teacher_model_type:
            case _:
                self.teacher_model = None

        if self.teacher_model is not None:
            for param in self.teacher_model.parameters():
                param.requires_grad = False  # fix teacher_model model
            self.teacher_model.eval()

        self.encode_task_layer = nn.Sequential(
            nn.Conv3d(conf.encoder.out_channels, conf.encoder.out_channels, 1),
            nn.Tanh(),
            nn.Conv3d(conf.encoder.out_channels, conf.embedding_dim, 1),
            # for quantize
        )
        if conf.project_decode:
            # self.decode_task_layer = nn.Sequential(
            #     nn.Conv3d(conf.decoder['out_dim'], decoder_config['out_dim'], 1),
            #     nn.Tanh(),
            #     nn.Conv3d(decoder_config['out_dim'], self.decoder_out_dim, 1),
            # )
            raise NotImplementedError
        else:
            self.decode_task_layer = nn.Identity()

        self.encode_task_layer.apply(init_linear_conv)
        self.decode_task_layer.apply(init_linear_conv)

    def get_tokens(self, data, **kwargs):
        z_q, token_ids, loss = self.encode(data)
        output = {'token': token_ids.view(data.shape[0], -1), 'input_img': data}

        return output

    def encode(self, x: torch.Tensor):
        encoder_features = self.encoder(x).feature_maps[-1]
        with torch.autocast(x.device, enabled=False):
            to_quantizer_features = self.encode_task_layer(encoder_features.type_as(self.encode_task_layer[-1].weight))
        return self.quantizer(to_quantizer_features)

    def decode(self, z_q: torch.Tensor, **kwargs):
        decoder_features = self.decoder(z_q)
        rec = self.decode_task_layer(decoder_features)
        return rec

    def get_codebook_indices(self, x, **kwargs):
        # for beit pre-training
        return self.get_tokens(x, **kwargs)['token']

    # return channel first following the image feature map convention
    def get_reconstruct_target(self, x, **kwargs):
        match self.conf.teacher_model_type:
            case _:
                return x

    def calculate_rec_loss(self, rec: torch.Tensor, target: torch.Tensor):
        rec = rec.view(*target.shape)
        match self.rec_loss_type:
            case 'cosine':
                rec = channel_last(rec).view(-1, rec.shape[1])
                target = channel_last(target).view(-1, target.shape[1])
                return 2 * torch.cosine_embedding_loss(rec, target, target=rec.new_ones(rec.shape[0]))
            case 'mse':
                if self.conf.norm_pix_loss:
                    raise NotImplementedError
                return torch_f.mse_loss(rec, target)
            case _:
                raise NotImplementedError

    def forward(self, x: torch.Tensor, **kwargs):
        target = self.get_reconstruct_target(x, **kwargs)

        z_q, token_ids, quant_loss = self.encode(x)
        x_rec = self.decode(z_q)

        rec_loss = self.calculate_rec_loss(x_rec, target)
        return {
            'quant_loss': quant_loss,
            'rec_loss': rec_loss,
            'loss': quant_loss + self.conf.beta * rec_loss,
        }

    def training_step(self, x: torch.Tensor, **kwargs: Any):
        log = self.forward(x)
        for k, v in log.items():
            self.log(f'train/{k}', v)

        return log

    def validation_step(self, x: torch.Tensor, **kwargs: Any):
        log = self.forward(x)
        for k, v in log.items():
            self.log(f'val/{k}', v)

        return log
