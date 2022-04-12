import pytorch_lightning as pl
from torchmetrics import AUROC
from tqdm import tqdm

from umei.datasets import Stoic2021Args, Stoic2021DataModule, Stoic2021Model
from umei.model import build_encoder
from umei.utils import MyWandbLogger, UMeIParser

def main():
    parser = UMeIParser((Stoic2021Args,), use_conf=True)
    args: Stoic2021Args = parser.parse_args_into_dataclasses()[0]
    test_dataloader = Stoic2021DataModule(args).test_dataloader()
    ensemble_output_dir = args.output_dir / 'ensemble'
    ensemble_output_dir.mkdir(parents=True, exist_ok=True)
    logger = MyWandbLogger(
        name=f'{args.exp_name}/ensemble',
        save_dir=str(ensemble_output_dir),
        group=args.exp_name,
    )
    encoder = build_encoder(args)
    model = Stoic2021Model(args, encoder)
    trainer = pl.Trainer(
        gpus=1,
        benchmark=True,
    )
    test_results = []
    num_models = 0

    def cal_auc(results):
        severity_auc = AUROC(pos_label=1)
        positivity_auc = AUROC(pos_label=1)

        for result, batch in zip(results, tqdm(test_dataloader)):
            positive_idx = batch[args.cls_key] >= 1
            if positive_idx.sum() > 0:
                severity_auc.update(result['severity_pred'][positive_idx],
                                    batch[args.cls_key][positive_idx] == 2)
            positivity_auc.update(result['positivity_pred'], batch[args.cls_key] >= 1)
        return {
            'auc-severity': severity_auc.compute(),
            'auc-positivity': positivity_auc.compute()
        }

    for run in range(1):
        for val_fold_id in range(3):
            output_dir = args.output_dir / f'fold{val_fold_id}' / f'run{run}'
            ckpt_path = list(output_dir.glob('auc-severity=*.ckpt'))[-1]
            # print(ckpt_path)
            test_outputs = trainer.predict(model, test_dataloader, ckpt_path=str(ckpt_path))
            print(cal_auc(test_outputs))

            for i, result in enumerate(test_outputs):
                if len(test_results) == i:
                    test_results.append(result)
                else:
                    for k, v in result.items():
                        test_results[i][k] += v
            num_models += 1

    for result in test_results:
        for k in ['severity_pred', 'positivity_pred']:
            result[k] /= num_models
    print(cal_auc(test_results))

if __name__ == '__main__':
    main()
