import einops
from einops import rearrange
import torch
from torch import distributed, nn
from torch.nn import functional as torch_f
import pytorch_lightning as pl

from umei.models.backbones.swin import SwinBackbone
from umei.utils import PathLike, channel_last

def ema_inplace(moving_avg: torch.Tensor, new: torch.Tensor, decay: float, norm: bool = False):
    moving_avg.data.mul_(decay).add_(new, alpha=1 - decay)
    if norm:
        moving_avg.data.copy_(l2norm(moving_avg.data))

def norm_ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha = (1 - decay))
    moving_avg.data.copy_(l2norm(moving_avg.data))

def l2norm(t: torch.Tensor):
    return torch_f.normalize(t, p=2, dim=-1)

def sample_vectors(data: torch.Tensor, num: int):
    n, device = data.shape[0], data.device

    if n >= num:
        indices = torch.randperm(n, device=device)[:num]
    else:
        indices = torch.randint(0, n, (num,), device=device)

    return data[indices]

def kmeans(data: torch.Tensor, num_clusters: int, num_iters: int = 10, use_cosine_sim: bool = True, init: str = 'random'):
    dim, dtype, device = data.shape[-1], data.dtype, data.device
    means = sample_vectors(data, num_clusters)

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = data @ means.T
        else:
            diffs = rearrange(data, 'n d -> n () d') - \
                    rearrange(means, 'c d -> () c d')
            dists = -(diffs ** 2).sum(dim=-1)

        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, einops.repeat(buckets, 'n -> n d', d=dim), data)
        new_means = new_means / bins_min_clamped[..., None]

        if use_cosine_sim:
            new_means = l2norm(new_means)

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins

class NormEMAVectorQuantizer(nn.Module):
    cluster_sizes: torch.Tensor

    def __init__(
        self,
        num_tokens: int,
        embedding_dim: int,
        beta: float,
        decay: float = 0.99,
        eps: float = 1e-5,
        statistic_code_usage: bool = True,
        kmeans_init: bool = False,
        codebook_init_path: PathLike | None = None,
    ):
        super().__init__()
        self.codebook_dim = embedding_dim
        self.num_tokens = num_tokens
        self.beta = beta
        self.decay = decay

        # learnable = True if orthogonal_reg_weight > 0 else False
        # self.embedding = EmbeddingEMA(self.num_tokens, self.codebook_dim, decay, eps, kmeans_init, codebook_init_path)

        if codebook_init_path is None:
            if kmeans_init:
                weight = torch.zeros(num_tokens, embedding_dim)
            else:
                weight = torch.randn(num_tokens, embedding_dim)
                weight = l2norm(weight)
            self.initialized = nn.Parameter(torch.tensor(not kmeans_init), requires_grad=False)
        else:
            print(f"load init codebook weight from {codebook_init_path}")
            codebook_ckpt_weight = torch.load(codebook_init_path, map_location='cpu')
            weight = codebook_ckpt_weight.clone()
            self.initialized = nn.Parameter(torch.tensor(True), requires_grad=False)
        self.embedding = nn.Parameter(weight, requires_grad=False)

        self.statistic_code_usage = statistic_code_usage
        if statistic_code_usage:
            self.register_buffer('cluster_sizes', torch.zeros(num_tokens))
        if distributed.is_available() and distributed.is_initialized():
            print("ddp is enable, so use ddp_reduce to sync the statistic_code_usage for each gpu!")
            self.all_reduce_fn = distributed.all_reduce
        else:
            self.all_reduce_fn = nn.Identity()

    def init_embed_(self, data: torch.Tensor):
        if self.initialized:
            return
        print("Performing kmeans init for codebook")
        centroids, cluster_sizes = kmeans(data, self.num_tokens, 10, use_cosine_sim=True)
        self.embedding.data.copy_(centroids)
        self.cluster_sizes.data.copy_(cluster_sizes)
        self.initialized.data.copy_(torch.Tensor([True]))

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        # z, 'b c h w -> b h w c'
        # z = rearrange(z, 'b c h w -> b h w c')
        z = channel_last(z)
        z = l2norm(z)
        z_flattened = z.reshape(-1, self.codebook_dim)
        # TODO: remove
        assert z.is_contiguous()

        self.init_embed_(z_flattened)
        # if not self.embedding.initialized:
        #     self.embedding.init_embed_(z_flattened)

        # d = z_flattened.pow(2).sum(dim=1, keepdim=True) + \
        #     self.embedding.weight.pow(2).sum(dim=1) - 2 * \
        #     torch.einsum('bd,nd->bn', z_flattened, self.embedding.weight)  # 'n d -> d n'

        cos_sim = z @ self.embedding.weight.T
        # token_ids = torch.argmin(d, dim=1)
        token_ids = cos_sim.argmax(dim=-1)
        # might be faster than indexing: https://github.com/pytorch/pytorch/issues/15245
        z_q = torch_f.embedding(token_ids, self.embedding)

        encodings = torch_f.one_hot(token_ids, self.num_tokens).type(z.dtype)

        if not self.training:
            with torch.no_grad():
                cluster_size = encodings.sum(0)
                self.all_reduce_fn(cluster_size)
                ema_inplace(self.cluster_sizes, cluster_size, self.decay)

        if self.training and self.embedding.update:
            # EMA cluster size

            bins = encodings.sum(0)
            self.all_reduce_fn(bins)

            # self.embedding.cluster_size_ema_update(bins)
            ema_inplace(self.cluster_sizes, bins, self.decay)

            zero_mask = (bins == 0)
            bins = bins.masked_fill(zero_mask, 1.)

            embed_sum = z_flattened.t() @ encodings
            self.all_reduce_fn(embed_sum)

            embed_normalized = (embed_sum / bins.unsqueeze(0)).t()
            embed_normalized = l2norm(embed_normalized)

            embed_normalized = torch.where(
                zero_mask[..., None], self.embedding.weight,
                embed_normalized
            )
            norm_ema_inplace(self.embedding, embed_normalized, self.decay)

        # compute loss for embedding
        loss = self.beta * F.mse_loss(z_q.detach(), z)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        # z_q, 'b h w c -> b c h w'
        z_q = rearrange(z_q, 'b h w c -> b c h w')
        return z_q, loss, token_ids


class VisualTokenizer(pl.LightningModule):
    def __init__(
        self,
        conf,
        num_tokens: int = 8192,
        embed_dim: int = 32,
        decay: float = 0.99,
        quantize_kmeans_init: bool = True,
        teacher_model_type: str | None = None,
        decoder_out_dim=512,
        rec_loss_type='cosine',
        **kwargs
    ):
        super().__init__()
        print(kwargs)

        # encoder & decode params
        print('Final encoder config', encoder_config := conf['encoder'])
        self.encoder = SwinBackbone(**encoder_config)

        print('Final decoder config', decoder_config := conf['decoder'])
        self.decoder = SwinDecoder(**decoder_config)

        self.quantize = NormEMAVectorQuantizer(
            num_tokens=num_tokens, embedding_dim=embed_dim, beta=1.0,
            kmeans_init=quantize_kmeans_init, decay=decay,
        )

        self.patch_size = encoder_config['patch_size']
        # self.token_shape = (
        #     encoder_config['img_size'] // self.patch_size,
        #     encoder_config['img_size'] // self.patch_size,
        # )

        ## Teacher model setting
        self.teacher_model_type = teacher_model_type
        self.decoder_out_dim = decoder_out_dim
        match self.teacher_model_type:
            case _:
                self.teacher_model = None

        if self.teacher_model is not None:
            for param in self.teacher_model.parameters():
                param.requires_grad = False  # fix teacher_model model
            self.teacher_model.eval()
            self.teacher_input_size = kwargs.get('teacher_input_size', 224)

        # task layer
        self.encode_task_layer = nn.Sequential(
            nn.Linear(encoder_config['embed_dim'], encoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(encoder_config['embed_dim'], embed_dim),
            # for quantize
        )
        self.decode_task_layer = nn.Sequential(
            nn.Linear(decoder_config['embed_dim'], decoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(decoder_config['embed_dim'], self.decoder_out_dim),
        )

        self.rec_loss_type = rec_loss_type

        self.logit_laplace_eps = 0.1
        self.kwargs = kwargs

        self.encode_task_layer.apply(self._init_weights)
        self.decode_task_layer.apply(self._init_weights)

    def _init_weights(self, m):
        match type(m):
            case nn.Linear:
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            case nn.LayerNorm:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'quantize.embedding.weight', 'decoder.cls_token', 'decoder.pos_embed', 'encoder.cls_token', 'encoder.pos_embed'}

    @property
    def device(self):
        return self.decoder.cls_token.device

    def get_tokens(self, data, **kwargs):

        data = self.pre_process(data)
        quantize, embed_ind, loss = self.encode(data)
        output = {}
        output['token'] = embed_ind.view(data.shape[0], -1)
        output['input_img'] = data

        return output

    def encode(self, x: torch.Tensor):
        encoder_features = self.encoder(x, return_patch_tokens=True)
        with torch.autocast(x.device, enabled=False):
            to_quantizer_features = self.encode_task_layer(encoder_features.type_as(self.encode_task_layer[-1].weight))
        quantize, loss, embed_ind = self.quantize(to_quantizer_features)
        return quantize, embed_ind, loss

    def decode(self, quantize, **kwargs):
        # reshape tokens to feature maps for patch embed in decoder
        # quantize = rearrange(quantize, 'b (h w) c -> b c h w', h=self.token_shape[0], w=self.token_shape[1])
        decoder_features = self.decoder(quantize, return_patch_tokens=True)
        rec = self.decode_task_layer(decoder_features)

        return rec

    def get_codebook_indices(self, x, **kwargs):
        # for beit pre-training
        return self.get_tokens(x, **kwargs)['token']

    def get_regress_target(self, x, **kwargs):
        match self.teacher_model_type:
            case _:
                return x

    def calculate_rec_loss(self, rec, target):
        if self.rec_loss_type == 'cosine':
            target = target / target.norm(dim=-1, keepdim=True)
            rec = rec / rec.norm(dim=-1, keepdim=True)
            rec_loss = (1 - (target * rec).sum(-1)).mean()
        else:
            raise NotImplementedError

        return rec_loss

    def forward(self, x: torch.Tensor, **kwargs):
        target = self.get_regress_target(x, **kwargs)

        quantize, embed_ind, emb_loss = self.encode(x)
        xrec = self.decode(quantize)

        rec_loss = self.calculate_rec_loss(xrec, target)
        loss = emb_loss + rec_loss

        log = {}
        split = "train" if self.training else "val"
        log[f'{split}/quant_loss'] = emb_loss.detach().mean()
        log[f'{split}/rec_loss'] = rec_loss.detach().mean()
        log[f'{split}/total_loss'] = loss.detach().mean()

        return loss, log
