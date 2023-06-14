from time import monotonic_ns

import numpy as np
import torch
from torch.nn import functional as nnf
from tqdm import trange

def main():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    torch.set_float32_matmul_precision('high')
    num_tokens = 8192
    num_samples = 1024
    d = 32

    errors = []
    for _ in range(100):
        index = torch.randint(num_tokens, size=(num_samples,))
        src = torch.randn(num_samples, d)
        out_scatter = torch.zeros(num_tokens, d).scatter_add_(0, index[:, None].expand(-1, d), src)
        out_ohmm = nnf.one_hot(index, num_tokens).to(src.dtype).T @ src
        errors.append((out_scatter - out_ohmm).mean().item())
    print(np.mean(errors))

    start = monotonic_ns()
    for _ in trange(30000):
        index = torch.randint(num_tokens, size=(num_samples,))
        src = torch.randn(num_samples, d)
        out_scatter = torch.zeros(num_tokens, d).scatter_add_(0, index[:, None].expand(-1, d), src)
    print('elapsed:', monotonic_ns() - start)

    start = monotonic_ns()
    for _ in trange(30000):
        index = torch.randint(num_tokens, size=(num_samples,))
        src = torch.randn(num_samples, d)
        out_ohmm = nnf.one_hot(index, num_tokens).to(src.dtype).T @ src
    print('elapsed:', monotonic_ns() - start)

if __name__ == '__main__':
    main()
