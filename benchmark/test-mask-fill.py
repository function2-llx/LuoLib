import torch
from tqdm import trange

def main():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    shape = (100, 100, 100)

    from time import monotonic_ns
    for _ in range(10000):
        x = torch.randn(shape)
        mask = torch.randint(2, size=shape).bool()
        x.masked_fill_(mask, 5)
        x[~mask] = 8

    start = monotonic_ns()
    for _ in trange(300000):
        x = torch.randn(shape)
        mask = torch.randint(2, size=shape).bool()
        x.masked_fill_(mask, 5)
    print('elapsed:', monotonic_ns() - start)

    start = monotonic_ns()
    for _ in trange(300000):
        x = torch.randn(shape)
        mask = torch.randint(2, size=shape).bool()
        x[mask] = 5
    print('elapsed:', monotonic_ns() - start)

if __name__ == '__main__':
    main()
