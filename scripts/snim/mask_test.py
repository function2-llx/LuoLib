import torch
from tqdm import trange

from umei.utils import UMeIParser
from umei.snim.args import MaskValue, SnimArgs
from umei.snim import SnimModel

def main():
    parser = UMeIParser(SnimArgs, use_conf=True)
    args: SnimArgs = parser.parse_args_into_dataclasses()[0]
    print(args)
    for value in MaskValue:
        if value == args.mask_value:
            print('mask value:', value)
    model = SnimModel(args)
    model = model.cuda()
    model = model.eval()
    torch.backends.cudnn.benchmark = True
    batch_size = args.train_batch_size
    print(f'expected mask ratio: {args.mask_ratio * 100}%')
    results = []
    with torch.no_grad():
        for _ in trange(1000, ncols=80):
            mask = model.gen_patch_mask(batch_size, args.sample_shape)
            mask = mask.view(batch_size, -1)
            results.append(mask.sum(dim=-1) * 100 / mask.shape[1])

    results = torch.cat(results)
    print(f'{results.shape[0]} samples mean±std: {results.mean().item():.3f}%±{results.std().item():.3f}%')

if __name__ == '__main__':
    main()
