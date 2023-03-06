from collections.abc import Sequence
from pathlib import Path

from einops import rearrange
from einops.layers.torch import Rearrange
import numpy as np
from pytorch_lightning.loggers import WandbLogger
import torch
from torch import distributed as dist, nn

from monai.data import MetaTensor
from monai.networks.layers import Act, get_act_layer
from monai.utils import ImageMetaKey

from umei.model_omega import ExpModelBase
from umei.models.backbones.swin import SwinBackbone
from umei.models.init import init_linear_conv3d
from umei.snim.omega import MaskValue, SnimConf
from umei.snim.utils import patchify, unpatchify
from umei.utils import channel_first, channel_last

class SnimEncoder(SwinBackbone):
    def __init__(self, conf: SnimConf):
        self.conf = conf
        super().__init__(**conf.backbone.kwargs)

        if conf.mask_value == MaskValue.PARAM:
            self.mask_token = nn.Parameter(torch.empty(1, 1, self.embed_dim))

    def apply_mask(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        conf = self.conf
        p_x = patchify(x.clone(), conf.mask_patch_size)
        match conf.mask_value:
            case MaskValue.UNIFORM:
                p_x[mask] = torch.rand(mask.sum(), p_x.shape[-1], device=x.device)
            case MaskValue.DIST:
                for i in range(x.shape[0]):
                    # force to use higher precision or nan will occur
                    with torch.autocast(x.device.type, enabled=False):
                        samples = rearrange(p_x[i], '... c -> c (...)').double()
                        mu = samples.mean(dim=1)
                        cov = samples.cov()
                        if torch.all(cov.count_nonzero(dim=0)):
                            dist = torch.distributions.MultivariateNormal(mu, cov)
                            sample = dist.sample(mask[i].sum().view(-1))
                            p_x[i][mask[i]] = sample.float()
                            if torch.isnan(sample).sum() > 0:
                                print(233)
            case MaskValue.PARAM:
                # nothing to do at this time, only for visualization
                p_x[mask] = 0
        x_mask = unpatchify(p_x, conf.mask_patch_size)
        return x_mask

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x_mask = self.apply_mask(x, mask)
        # when using mask_token, original x is input into patch embedding
        if self.conf.mask_value != MaskValue.PARAM:
            x = x_mask
        x = self.patch_embed(x)
        if self.conf.mask_value == MaskValue.PARAM:
            x = channel_last(x)
            x[mask] = self.mask_token
            x = channel_first(x)
        feature_maps = self.forward_layers(x)
        return x_mask, feature_maps

class SnimDecoder(nn.Module):
    def __init__(
        self,
        conf: SnimConf,
        layer_channels: list[int],
        lateral: bool = True,
        act: str | tuple = Act.GELU,
    ):
        super().__init__()
        self.conf = conf
        num_layers = len(layer_channels)

        self.lateral_projects = nn.ModuleList([
            nn.Sequential(
                nn.Linear(layer_channels[i], layer_channels[i]),
                nn.LayerNorm(layer_channels[i]),
            ) if lateral
            else nn.Identity()
            for i in range(num_layers - 1)
        ])

        self.up_projects = nn.ModuleList([
            nn.Sequential(
                nn.Linear(layer_channels[i + 1], 8 * layer_channels[i]),
                Rearrange('n h w d (s0 s1 s2 c) -> n (h s0) (w s1) (d s2) c', s0=2, s1=2, s2=2),
                nn.LayerNorm(layer_channels[i]),
            )
            for i in range(num_layers - 1)
        ])

        self.acts = nn.ModuleList([
            get_act_layer(act)
            for _ in range(num_layers - 1)
        ])

        self.projects = nn.ModuleList([
            nn.Sequential(
                nn.Linear(layer_channels[i], layer_channels[i]),
                nn.LayerNorm(layer_channels[i]),
                get_act_layer(act),
            )
            for i in range(num_layers - 1)
        ])
        self.reg_head = nn.Sequential(
            nn.Linear(layer_channels[0], np.product(conf.mask_patch_size) * conf.num_input_channels),
            Rearrange('n h w d (s0 s1 s2 c) -> n c (h s0) (w s1) (d s2)', **{
                f's{i}': size
                for i, size in enumerate(conf.mask_patch_size)
            }),
        )
        self.dis_head = nn.Sequential(
            nn.Linear(layer_channels[0], 1),
            Rearrange('... 1 -> ...'),
        )

        self.apply(init_linear_conv3d)

    def forward(self, feature_maps: list[torch.Tensor]):
        x = channel_last(feature_maps[-1]).contiguous()
        for z, lateral_proj, up_proj, act, proj in zip(
            feature_maps[-2::-1],
            self.lateral_projects[::-1],
            self.up_projects[::-1],
            self.acts[::-1],
            self.projects[::-1],
        ):
            z = channel_last(z).contiguous()
            z = lateral_proj(z)
            x = up_proj(x)
            x = act(x + z)
            x = proj(x)

        reg_pred = self.reg_head(x)
        dis_pred = self.dis_head(x)
        return reg_pred, dis_pred

class SnimModel(ExpModelBase):
    logger: WandbLogger

    def __init__(self, conf: SnimConf):
        super().__init__(conf)
        self.conf = conf

        self.corner_counter = nn.Conv3d(1, 1, kernel_size=conf.p_block_shape, bias=False)
        self.corner_counter.weight.requires_grad = False

        self.encoder = SnimEncoder(conf)
        self.decoder = SnimDecoder(conf, **conf.decoder.kwargs)
        self.reg_loss_fn = {
            'l1': nn.L1Loss(),
            'l2': nn.MSELoss(),
        }[conf.loss]
        self.dis_loss_fn = nn.BCEWithLogitsLoss()

        self.initialize_weights()

    def initialize_weights(self):
        if self.conf.mask_value == MaskValue.PARAM:
            torch.nn.init.normal_(self.encoder.mask_token, std=0.02)
        nn.init.constant_(self.corner_counter.weight, 1)

    # def _init_weights(self, m: nn.Module):
    #     if isinstance(m, nn.Linear):
    #         # we use xavier_uniform following official JAX ViT:
    #         torch.nn.init.xavier_uniform_(m.weight)
    #         if m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.weight, 1.0)
    #         nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)) and all(k <= s for k, s in zip(m.kernel_size, m.stride)):
    #         w: torch.Tensor = m.weight.data
    #         torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def gen_patch_mask(self, batch_size: int, img_shape: Sequence[int], device=None) -> torch.Tensor:
        mask_shape = [
            size // patch_size
            for size, patch_size in zip(img_shape, self.conf.mask_patch_size)
        ]
        # corner spatial shape
        corner_ss = [
            size + block_patch_num - 1
            for size, block_patch_num in zip(mask_shape, self.conf.p_block_shape)
        ]
        sample_space_size = np.product(corner_ss)
        if self.conf.mask_ratio == 1:
            mask_num = np.product(mask_shape)
        else:
            sample_num = np.round(
                np.log(1 - self.conf.mask_ratio) /
                np.log(1 - np.product(self.conf.p_block_shape) / sample_space_size)
            )
            mask_num = int(sample_space_size * (1 - (1 - 1 / sample_space_size) ** sample_num))
        if mask_num == 0:
            mask = torch.zeros(batch_size, *mask_shape, dtype=torch.bool)
        else:
            noise: torch.Tensor = torch.rand(batch_size, sample_space_size, device=self.device)
            kth = noise.kthvalue(mask_num, dim=-1, keepdim=True).values
            corner_mask = rearrange(
                noise <= kth,
                'n (h w d) -> n 1 h w d',
                h=corner_ss[0], w=corner_ss[1], d=corner_ss[2],
            )
            mask = self.corner_counter(corner_mask.float()).round() >= 1
            mask = rearrange(mask, 'n 1 ... -> n ...')

        return mask

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        # p_xxx: patchified xxx
        conf = self.conf
        p_x = patchify(x, conf.mask_patch_size)
        if mask is None:
            mask = self.gen_patch_mask(x.shape[0], x.shape[2:])
        x_mask, feature_maps = self.encoder.forward(x, mask)
        reg_pred, dis_logit = self.decoder.forward(feature_maps)
        p_reg_pred = patchify(reg_pred, conf.mask_patch_size)
        if conf.norm_pix_loss:
            # note: the mean and var are calculated across channels
            mean = p_x.mean(dim=-1, keepdim=True)
            var = p_x.var(dim=-1, keepdim=True)
            p_x = (p_x - mean) / torch.sqrt(var + 1e-6)
        reg_loss_masked = self.reg_loss_fn(p_reg_pred[mask], p_x[mask])
        reg_loss_visible = self.reg_loss_fn(p_reg_pred[~mask], p_x[~mask])
        reg_loss = reg_loss_masked + conf.reg_loss_visible_factor * reg_loss_visible
        dis_loss = self.dis_loss_fn(dis_logit, mask.float())
        loss = reg_loss + conf.dis_loss_factor * dis_loss
        if torch.isnan(loss):
            print(233)
        return reg_loss_masked, reg_loss_visible, reg_loss, dis_loss, loss, mask, x_mask, reg_pred, dis_logit

    def training_step(self, x: MetaTensor, *args, **kwargs):
        reg_loss_masked, reg_loss_visible, reg_loss, dis_loss, loss, *_ = self.forward(x.as_tensor())
        self.log('train/reg-loss(masked)', reg_loss_masked)
        self.log('train/reg-loss(visible)', reg_loss_visible)
        self.log('train/dis-loss', dis_loss)
        self.log('train/loss', loss)
        return loss

    def validation_step(self, x: MetaTensor, *args, **kwargs):
        filename = Path(x.meta[ImageMetaKey.FILENAME_OR_OBJ][0]).name
        x = x.as_tensor()
        reg_loss_masked, reg_loss_visible, reg_loss, dis_loss, loss, mask, x_mask, reg_pred, dis_logit = self.forward(x)
        self.log('val/reg-loss(masked)', reg_loss_masked, sync_dist=True)
        self.log('val/reg-loss(visible)', reg_loss_visible, sync_dist=True)
        self.log('val/dis-loss', dis_loss, sync_dist=True)
        self.log('val/loss', loss, sync_dist=True)

        dis_pred = dis_logit.sigmoid() > 0.5

        tot = mask.numel()
        s_mask = mask.sum()
        s_pred = dis_pred.sum()
        tp = (dis_pred & mask).sum()
        tn = (~(dis_pred | mask)).sum()
        self.log('val/dis-acc', (dis_pred == mask).sum() / tot * 100, sync_dist=True)
        self.log('val/mask-precision', tp / s_pred * 100 if s_pred > 0 else int(s_mask == 0), sync_dist=True)
        self.log('val/mask-recall', tp / s_mask * 100 if s_mask > 0 else 1, sync_dist=True)
        self.log('val/visible-precision', tn / (tot - s_pred) * 100 if s_pred > 0 else int(s_mask == 0), sync_dist=True)
        self.log('val/visible-recall', tn / (tot - s_mask) * 100 if s_mask > 0 else 1, sync_dist=True)

        conf = self.conf
        pred_ol = patchify(reg_pred.clone(), conf.mask_patch_size)
        pred_ol[~mask] = patchify(x, conf.mask_patch_size)[~mask].to(pred_ol.dtype)
        pred_ol = unpatchify(pred_ol, conf.mask_patch_size)

        slice_ids = [int(x.shape[-1] * r) for r in [0.25, 0.5, 0.75]]
        if self.trainer.world_size > 1:
            filenames = [None] * self.trainer.world_size
            dist.all_gather_object(filenames, filename)
            flatten = Rearrange('w n ... -> (w n) ...')
            all_x, all_x_mask, all_pred, all_pred_ol = map(flatten, self.all_gather([x, x_mask, reg_pred, pred_ol]))
        else:
            filenames = [filename]
            all_x, all_x_mask, all_pred, all_pred_ol = x, x_mask, reg_pred, pred_ol

        if self.trainer.is_global_zero:
            for filename, x, x_mask, reg_pred, pred_ol in zip(filenames, all_x, all_x_mask, all_pred, all_pred_ol):
                for slice_idx in slice_ids:
                    self.logger.log_image(
                        f'val/{filename}/{slice_idx}',
                        images=[
                            x[..., slice_idx].float().rot90(dims=(1, 2)).cpu(),
                            x_mask[..., slice_idx].float().rot90(dims=(1, 2)).cpu(),
                            reg_pred[..., slice_idx].float().rot90(dims=(1, 2)).cpu(),
                            pred_ol[..., slice_idx].float().rot90(dims=(1, 2)).cpu(),
                        ],
                        caption=['original', 'mask', 'pred', 'pred-ol'],
                    )


    def configure_optimizers(self):
        optim_config = super().configure_optimizers()
        optim_config.pop('monitor')

        return optim_config
