import torch

from luolib.models import MaskedAttentionDecoder

def main():
    bs = 2
    sp = 3
    feature_channels = 256
    num_attention_heads = 8
    pixel_embedding_dim = 32
    num_queries = 2
    decoder = MaskedAttentionDecoder(
        sp, feature_channels, num_attention_heads, pixel_embedding_dim, num_queries=num_queries
    ).cuda()
    spatial_shapes = [
        # (48, 192, 192),
        (6,) * 3,
        (12,) * 3,
        (24, ) * 3,
    ]
    feature_maps = [
        torch.randn(bs, feature_channels, *shape, device='cuda')
        for shape in spatial_shapes
    ]
    pixel_embedding_shape = (48, 192, 192)
    pixel_embedding = torch.randn(bs, pixel_embedding_dim, *pixel_embedding_shape, device='cuda')
    res = decoder.forward(feature_maps, pixel_embedding)
    print(233)

if __name__ == '__main__':
    main()
