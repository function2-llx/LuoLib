import torch
from tqdm import trange

from umei.snim.args import MaskValue, SnimArgs
from umei.snim import SnimModel

def main():
    args: SnimArgs = SnimArgs.from_yaml_file('conf/snim/mask-test.yml')
    print(args)
    for value in MaskValue:
        if value == args.mask_value:
            print('mask value:', value)
    model = SnimModel(args).cuda().eval()
    torch.backends.cudnn.benchmark = True
    batch_size = args.train_batch_size
    print('expected mask ratio:', args.mask_ratio)
    results = []
    with torch.no_grad():
        for _ in trange(1000, ncols=80):
            mask = model.gen_patch_mask(batch_size, args.sample_shape)
            mask = mask.view(batch_size, -1)
            results.append(mask.sum(dim=-1) / mask.shape[1])

    results = torch.cat(results)
    print(f'{results.mean().item()}±{results.std().item()}')

if __name__ == '__main__':
    main()
