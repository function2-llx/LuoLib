from collections.abc import Sequence

from einops import rearrange
import pytorch_lightning as pl
import torch
from torch import distributed, nn
from torch.nn import functional as torch_f

from umei.models.adaptive_resampling import AdaptiveUpsampling
from umei.models.backbones.swin import SwinBackbone, SwinLayer
from umei.models.layers import LayerNormNd, Norm
from umei.utils import PathLike, channel_first, channel_last

class SwinVQDecoder(nn.Module):
    def __init__(
        self,
        layer_channels: Sequence[int],
        kernel_sizes: Sequence[int | Sequence[int]],
        layer_depths: Sequence[int],
        num_heads: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        num_layers = len(layer_depths)
        if isinstance(layer_channels, int):
            layer_channels = [layer_channels << i for i in range(num_layers)]

        self.layers = nn.ModuleList([
            SwinLayer(
                layer_channels[i],
                layer_depths[i],
                num_heads[i],
                _max_window_size := kernel_sizes[i],
                _drop_path := 0,
                mlp_ratio,
                qkv_bias,
                drop_rate,
                attn_drop_rate,
                Norm.LAYER,
                use_checkpoint,
            )
            for i in range(num_layers)
        ])

        self.upsamplings = nn.ModuleList([
            AdaptiveUpsampling(
                layer_channels[i + 1],
                layer_channels[i],
            )
            for i in range(num_layers - 1)
        ])

        self.norms = nn.ModuleList([
            LayerNormNd(layer_channels[i])
            for i in range(num_layers)
        ])

        self.apply(init_linear_conv)

    def no_weight_decay(self):
        ret = set()
        for name, _ in self.named_parameters():
            if 'relative_position_bias_table' in name:
                ret.add(name)
        return ret

    def forward(self, x: torch.Tensor, target_z_size: int):
        for layer, norm, upsampling in zip(self.layers[::-1], self.norms[::-1], self.upsamplings[::-1]):
            x = layer(x)
            x = norm(x)
            x = upsampling(x, x.shape[-1] < target_z_size)
        x = self.layers[0](x)
        x = self.norms[0](x)
        return x

def ema_inplace(moving_avg: torch.Tensor, new: torch.Tensor, decay: float, norm: bool = False):
    moving_avg.mul_(decay).add_(new, alpha=1 - decay)
    if norm:
        # FIXME: divide inplace? This results in two copies
        moving_avg.copy_(l2norm(moving_avg))

def l2norm(t: torch.Tensor):
    return torch_f.normalize(t, p=2, dim=-1)

def sample_vectors(data: torch.Tensor, num: int):
    n, device = data.shape[0], data.device

    if n >= num:
        indices = torch.randperm(n, device=device)[:num]
    else:
        indices = torch.randint(0, n, (num,), device=device)

    return data[indices]

def init_linear_conv(m: nn.Module):
    match type(m):
        case nn.Linear | nn.Conv3d:
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

# if use_cosine_sim, data must be l2-normalized
def kmeans(
    data: torch.Tensor, num_clusters: int, num_iters: int = 10, use_cosine_sim: bool = True, init: str = 'random'
):
    match init:
        case 'random':
            means = sample_vectors(data, num_clusters)
        case _:
            raise NotImplementedError
    assert num_iters > 0
    cluster_sizes = None  # PyCharm good

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = data @ means.T
        else:
            diffs = rearrange(data, 'n d -> n () d') - \
                    rearrange(means, 'c d -> () c d')
            dists = -(diffs ** 2).sum(dim=-1)

        cluster_ids = dists.max(dim=-1).indices
        cluster_sizes = torch.bincount(cluster_ids, minlength=num_clusters)
        means.scatter_reduce(
            dim=0,
            index=cluster_ids[:, None].expand(-1, data.shape[-1]),
            src=data,
            reduce='mean',
            include_self=False,
        )

    return means, cluster_sizes

class NormEMAVectorQuantizer(nn.Module):
    code_usage: torch.Tensor
    initialized: torch.BoolTensor

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        beta: float,    # commitment loss factor
        decay: float = 0.99,
        track_code_usage: bool = True,
        kmeans_init: bool = False,
        codebook_init_path: PathLike | None = None,
    ):
        super().__init__()
        self.num_tokens = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.decay = decay

        if codebook_init_path is None:
            if kmeans_init:
                weight = torch.zeros(num_embeddings, embedding_dim)
                initialized = False
            else:
                weight = torch.randn(num_embeddings, embedding_dim)
                weight = l2norm(weight)
                initialized = True
        else:
            print(f"load init codebook weight from {codebook_init_path}")
            codebook_ckpt_weight = torch.load(codebook_init_path, map_location='cpu')
            weight = codebook_ckpt_weight.clone()
            initialized = True
        self.register_buffer('initialized', torch.tensor(initialized))
        self.embedding = nn.Parameter(weight, requires_grad=False)

        self.statistic_code_usage = track_code_usage
        if track_code_usage:
            self.register_buffer('code_usage', torch.zeros(num_embeddings))
        if distributed.is_available() and distributed.is_initialized():
            print("ddp is enable, so use ddp_reduce to sync the statistic_code_usage for each gpu!")
            self.all_reduce_fn = distributed.all_reduce
        else:
            self.all_reduce_fn = nn.Identity()

    def init_embed_(self, data: torch.Tensor):
        print("Performing kmeans init for codebook")
        centroids, cluster_sizes = kmeans(data, self.num_tokens, 10, use_cosine_sim=True)
        self.embedding.copy_(centroids)
        self.code_usage.copy_(cluster_sizes)
        self.initialized.fill_(True)

    def forward(self, z):
        z = channel_last(z).contiguous()
        z = l2norm(z)
        if not self.initialized:
            self.init_embed_(z.view(-1, self.embedding_dim))

        cos_sim = z @ self.embedding.weight.T
        embed_ids = cos_sim.argmax(dim=-1)
        flat_embed_ids = embed_ids.view(-1)
        # might be faster than indexing: https://github.com/pytorch/pytorch/issues/15245
        z_q = torch_f.embedding(embed_ids, self.embedding)

        code_usage = torch.bincount(flat_embed_ids, minlength=self.num_tokens)
        self.all_reduce_fn(code_usage)
        ema_inplace(self.code_usage, code_usage, self.decay)

        if self.training:
            embed_mean = self.embedding.scatter_reduce(
                dim=0,
                index=flat_embed_ids[:, None].expand(-1, self.embedding_dim),
                src=z,
                reduce='mean',
                include_self=False,
            )
            embed_mean = l2norm(embed_mean)
            ema_inplace(self.embedding, embed_mean, self.decay, norm=True)

        # compute loss for embedding
        # loss = self.beta * torch_f.mse_loss(z_q.detach(), z)
        loss = self.beta * 2 * torch.cosine_embedding_loss(
            z_q.view(-1, self.embedding_dim).detach(),
            z.view(-1, self.embedding_dim),
            target=torch.ones_like(flat_embed_ids),
        )

        # straight-through gradient
        z_q = z + (z_q - z).detach()
        z_q = channel_first(z_q)
        return z_q, embed_ids, loss

class VisualTokenizer(pl.LightningModule):
    def __init__(
        self,
        conf,
        num_tokens: int = 8192,
        embedding_dim: int = 32,
        decay: float = 0.99,
        beta: float = 1.,
        quantize_kmeans_init: bool = True,
        teacher_model_type: str | None = None,
        decoder_out_dim: int = 512,
        rec_loss_type: str = 'cosine',
        track_code_usage: bool = True,
        codebook_init_path: PathLike | None = None,
        **kwargs,
    ):
        super().__init__()
        print(kwargs)

        # encoder & decode params
        print('Final encoder config', encoder_config := conf['encoder'])
        self.encoder = SwinBackbone(**encoder_config)

        print('Final decoder config', decoder_config := conf['decoder'])
        self.decoder = SwinVQDecoder(**decoder_config)

        self.quantizer = NormEMAVectorQuantizer(
            num_tokens,
            embedding_dim,
            beta,
            decay,
            track_code_usage,
            quantize_kmeans_init,
            codebook_init_path,
        )

        self.patch_size = encoder_config['patch_size']

        # Teacher model setting
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
            nn.Conv3d(encoder_config['embed_dim'], encoder_config['embed_dim'], 1),
            nn.Tanh(),
            nn.Conv3d(encoder_config['embed_dim'], embedding_dim, 1),
            # for quantize
        )
        self.decode_task_layer = nn.Sequential(
            nn.Conv3d(decoder_config['embed_dim'], decoder_config['embed_dim'], 1),
            nn.Tanh(),
            nn.Conv3d(decoder_config['embed_dim'], self.decoder_out_dim, 1),
        )

        self.rec_loss_type = rec_loss_type

        self.encode_task_layer.apply(init_linear_conv)
        self.decode_task_layer.apply(init_linear_conv)

    def no_weight_decay(self):
        return {'quantize.embedding'}

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

    # return channel first!
    def get_reconstruct_target(self, x, **kwargs):
        match self.teacher_model_type:
            case _:
                return x

    def calculate_rec_loss(self, rec: torch.Tensor, target: torch.Tensor):
        match self.rec_loss_type:
            case 'cosine':
                rec = channel_last(rec).view(-1, rec.shape[1])
                target = channel_last(target).view(-1, target.shape[1])
                return torch.cosine_embedding_loss(rec, target, target=rec.new_ones(rec.shape[0]))
            case 'mse':
                return torch_f.mse_loss(rec, target)
            case _:
                raise NotImplementedError

    def forward(self, x: torch.Tensor, **kwargs):
        target = self.get_reconstruct_target(x, **kwargs)

        z_q, token_ids, quant_loss = self.encode(x)
        x_rec = self.decode(z_q)

        rec_loss = self.calculate_rec_loss(channel_last(x_rec).contiguous(), target)
        loss = quant_loss + rec_loss

        log = {}
        split = "train" if self.training else "val"
        log[f'{split}/quant_loss'] = quant_loss.detach().mean()
        log[f'{split}/rec_loss'] = rec_loss.detach().mean()
        log[f'{split}/total_loss'] = loss.detach().mean()

        return loss, log
