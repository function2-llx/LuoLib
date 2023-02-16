import os

import numpy as np
import torch
from tqdm import trange

from monai.umei import BackboneOutput
from umei.models.swin_monai import SwinTransformer
from umei.models.swin import SwinBackbone, compute_relative_position_index

def test_layer():
    print('test layer')
    from umei.models.swin import SwinLayer
    from monai.networks.nets.swin_unetr import BasicLayer

    stage = 2
    batch_size = 64
    dim = 48 << stage
    shape = (batch_size, dim, 48 >> stage, 48 >> stage, 48 >> stage)
    depth = 12
    num_heads = 3 << stage
    window_size = [6, 6, 6]
    print(f'stage: {stage}, dim: {dim}, depth: {depth}, num_heads: {num_heads}')
    print(shape)
    print('window:', window_size)

    models = {
        'my': SwinLayer(
            dim,
            depth,
            num_heads,
            window_size,
        ),
        'monai': BasicLayer(
            dim,
            depth,
            num_heads,
            window_size,
            drop_path=[.0] * depth,
        ),
    }

    # warm up
    for i in range(30):
        x = torch.randn(shape)
        for model in models.values():
            out: BackboneOutput = model(x)
            # if not i:
            #     for feature_map in out.feature_maps:
            #         print(feature_map.shape)

    for _ in range(3):
        for name, model in models.items():
            from time import monotonic_ns
            start = monotonic_ns()
            for _ in trange(100, desc=name, ncols=80):
                x = torch.randn(shape)
                model(x)
            print('elapsed:', monotonic_ns() - start)


def test_wa():
    print('test wa')
    from umei.models.swin import WindowAttention as MyWA
    from monai.networks.nets.swin_unetr import WindowAttention

    stage = 3
    test_num = 1000
    batch_size = 128
    dim = 48 << stage
    num_heads = 3 << stage
    window_size = (6, 6, 6)
    shape = (batch_size * np.power(48 >> stage, 3) // np.prod(window_size), np.prod(window_size), dim)
    print(f'stage: {stage}, dim: {dim}, num_heads: {num_heads}')
    print(shape)
    print('window:', window_size)
    relative_position_index = compute_relative_position_index(window_size)
    models = {
        'my': MyWA(
            dim,
            num_heads,
            window_size,
            relative_position_index,
        ),
        'monai': WindowAttention(
            dim,
            num_heads,
            window_size,
        ),
    }

    # warm up
    for i in range(30):
        x = torch.randn(shape)
        for model in models.values():
            out: BackboneOutput = model(x, None)
            # if not i:
            #     for feature_map in out.feature_maps:
            #         print(feature_map.shape)

    for _ in range(2):
        for name, model in models.items():
            from time import monotonic_ns
            start = monotonic_ns()
            for _ in trange(test_num, desc=name, ncols=80):
                x = torch.randn(shape)
                model(x, None)
            print('elapsed:', monotonic_ns() - start)

def test_block():
    print('test block')
    from umei.models.swin import SwinTransformerBlock as MyBlock
    from monai.networks.nets.swin_unetr import SwinTransformerBlock

    stage = 3
    test_num = 1000
    batch_size = 128
    dim = 48 << stage
    num_heads = 3 << stage
    window_size = (6, 6, 6)
    shift_size = [0, 0, 0]
    shape = (batch_size, 48 >> stage, 48 >> stage, 48 >> stage, dim)
    # shape = (batch_size * np.power(48 >> stage, 3) // np.prod(window_size), np.prod(window_size), dim)
    print(f'stage: {stage}, dim: {dim}, num_heads: {num_heads}')
    print(shape)
    print('window:', window_size)
    relative_position_index = compute_relative_position_index(window_size)

    models = {
        'my': MyBlock(
            dim,
            num_heads,
            window_size,
            relative_position_index,
        ),
        'monai': SwinTransformerBlock(
            dim,
            num_heads,
            window_size,
            shift_size=[0, 0, 0],
        ),
    }
    kwargs = {
        'my': {
            'window_size': window_size,
            'shift_size': shift_size,
            'attn_mask': None,
        },
        'monai': {
            'mask_matrix': None,
        }
    }

    # warm up
    for i in range(30):
        x = torch.randn(shape)
        for name, model in models.items():
            out = model(x, **kwargs[name])

    for _ in range(2):
        for name, model in models.items():
            from time import monotonic_ns
            start = monotonic_ns()
            for _ in trange(test_num, desc=name, ncols=80):
                x = torch.randn(shape)
                model(x, **kwargs[name])
            print('elapsed:', monotonic_ns() - start)

def test_model():
    patch_size = 4
    shape = (4, 1, 48 * patch_size, 48 * patch_size, 48 * patch_size)
    models = {
        'my': SwinBackbone(
            in_channels=1,
            layer_channels=[96, 192, 384, 768],
            kernel_sizes=[6, 6, 6, 6],
            layer_depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            stem_stride=4,
            stem_channels=48,
        ),
        'monai': SwinTransformer(
            in_chans=1,
            embed_dim=96,
            window_size=[6, 6, 6],
            patch_size=[patch_size] * 3,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
        ),
    }

    print('shape:', shape)


    # warm up
    for i in range(10):
        x = torch.randn(shape)
        for name, model in models.items():
            out: BackboneOutput = model(x)
            if not i:
                print(f'warmup {name}')
                for feature_map in out.feature_maps:
                    print(feature_map.shape)

    for _ in range(3):
        for name, model in models.items():
            from time import monotonic_ns
            start = monotonic_ns()
            for _ in trange(100, desc=name, ncols=80):
                x = torch.randn(shape)
                model(x)
            print('elapsed:', monotonic_ns() - start)

def main():
    print('cuda:', os.environ['CUDA_VISIBLE_DEVICES'])
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    print(torch.get_float32_matmul_precision())
    torch.set_float32_matmul_precision('high')
    print(torch.get_float32_matmul_precision())

    from torch import nn

    # test_wa()
    # test_block()
    # test_layer()
    test_model()

if __name__ == '__main__':
    main()
