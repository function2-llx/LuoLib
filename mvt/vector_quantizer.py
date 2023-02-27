from __future__ import annotations

from einops import rearrange
import torch
from torch import distributed, nn
from torch.nn import functional as torch_f

from umei.utils import PathLike, channel_first, channel_last

class NormEMAVectorQuantizer(nn.Module):
    code_usage: torch.Tensor
    initialized: torch.BoolTensor

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        decay: float = 0.99,
        track_code_usage: bool = True,
        kmeans_init: bool = True,
        codebook_init_path: PathLike | None = None,
    ):
        super().__init__()
        self.num_tokens = num_embeddings
        self.embedding_dim = embedding_dim
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
            weight = torch.load(codebook_init_path, map_location='cpu')
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
            self.all_reduce_fn = lambda x: x

    def init_embed_(self, data: torch.Tensor):
        print("Performing kmeans init for codebook")
        centroids, cluster_sizes = kmeans(data, self.num_tokens, 10, use_cosine_sim=True)
        self.embedding.copy_(centroids)
        self.code_usage.copy_(cluster_sizes)
        self.initialized.fill_(True)

    def forward(self, z: torch.Tensor):
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
        loss = torch.cosine_embedding_loss(
            z_q.view(-1, self.embedding_dim).detach(),
            z.view(-1, self.embedding_dim),
            target=torch.ones_like(flat_embed_ids),
        )

        # straight-through gradient
        z_q = z + (z_q - z).detach()
        z_q = channel_first(z_q)
        return z_q, embed_ids, loss

def ema_inplace(moving_avg: torch.Tensor, new: torch.Tensor, decay: float, norm: bool = False):
    moving_avg.mul_(decay).add_(new, alpha=1 - decay)
    if norm:
        # FIXME: divide inplace? This results in two copies
        moving_avg.copy_(l2norm(moving_avg))

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
            sim = data @ means.T
        else:
            diff = rearrange(data, 'n d -> n () d') - \
                    rearrange(means, 'c d -> () c d')
            sim = -(diff ** 2).sum(dim=-1)

        cluster_ids = sim.max(dim=-1).indices
        cluster_sizes = torch.bincount(cluster_ids, minlength=num_clusters)
        means.scatter_reduce(
            dim=0,
            index=cluster_ids[:, None].expand(-1, data.shape[-1]),
            src=data,
            reduce='mean',
            include_self=False,
        )

    return means, cluster_sizes

def sample_vectors(data: torch.Tensor, num: int):
    n, device = data.shape[0], data.device

    if n >= num:
        indices = torch.randperm(n, device=device)[:num]
    else:
        indices = torch.randint(0, n, (num,), device=device)

    return data[indices]

def l2norm(t: torch.Tensor):
    return torch_f.normalize(t, p=2, dim=-1)