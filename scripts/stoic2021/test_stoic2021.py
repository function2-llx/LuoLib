from copy import deepcopy
from dataclasses import dataclass, field

import pytorch_lightning as pl
from torchmetrics import AUROC
from tqdm import tqdm

from umei.datasets import Stoic2021Args, Stoic2021DataModule, Stoic2021Model
from umei.model import build_encoder
from umei.utils import UMeIParser

# torch.multiprocessing.set_sharing_strategy('file_system')

@dataclass
class Stoic2021InferArgs(Stoic2021Args):
    use_origin_size: bool = field(default=False)

def main():
    parser = UMeIParser((Stoic2021InferArgs,), use_conf=True)
    args: Stoic2021InferArgs = parser.parse_args_into_dataclasses()[0]
    print(args)
    args.dataloader_num_workers = 10
    args.per_device_eval_batch_size = 1
    all_infer_args = [args]
    if args.use_origin_size:
        args_origin_size = deepcopy(args)
        args_origin_size.sample_size = args_origin_size.sample_slices = -1
        all_infer_args.append(args_origin_size)

    ensemble_output_dir = args.output_dir / 'ensemble'
    ensemble_output_dir.mkdir(parents=True, exist_ok=True)
    encoder = build_encoder(args)
    model = Stoic2021Model(args, encoder)
    trainer = pl.Trainer(
        logger=False,
        # logger=MyWandbLogger(
        #     name=f'{args.exp_name}/ensemble',
        #     save_dir=str(ensemble_output_dir),
        #     group=args.exp_name,
        # ),
        gpus=1,
        benchmark=True,
    )

    def cal_auc(results):
        severity_auc = AUROC(pos_label=1)
        positivity_auc = AUROC(pos_label=1)

        for result, batch in zip(results, tqdm(Stoic2021DataModule(args).test_dataloader())):
            positive_idx = batch[args.cls_key] >= 1
            if positive_idx.sum() > 0:
                severity_auc.update(result['severity_pred'][positive_idx],
                                    batch[args.cls_key][positive_idx] == 2)
            positivity_auc.update(result['positivity_pred'], batch[args.cls_key] >= 1)
        return {
            'auc-severity': severity_auc.compute(),
            'auc-positivity': positivity_auc.compute()
        }

    test_results = []
    num_models = 0

    def update(results: list[dict], i: int, result: dict):
        if len(results) == i:
            results.append(result)
        else:
            for k, v in result.items():
                results[i][k] += v

    def reduce(results: list[dict], n: int):
        for result in results:
            for k in ['severity_pred', 'positivity_pred']:
                result[k] /= n

    for run in range(1):
        for val_fold_id in range(9):
            output_dir = args.output_dir / f'fold{val_fold_id}' / f'run{run}'
            ckpt_path = list(output_dir.glob('auc-severity=*.ckpt'))[-1]
            cur_test_results = []
            for infer_args in all_infer_args:
                test_outputs = trainer.predict(model, Stoic2021DataModule(infer_args).test_dataloader(), ckpt_path=str(ckpt_path))
                for i, result in enumerate(test_outputs):
                    update(cur_test_results, i, result)
                    update(test_results, i, result)

                num_models += 1

            reduce(cur_test_results, len(all_infer_args))
            print(cal_auc(cur_test_results))

    reduce(test_results, num_models)
    print(cal_auc(test_results))

if __name__ == '__main__':
    main()
