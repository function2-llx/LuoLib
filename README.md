# UMeI

U-shaped network for Medical Imaging

## Setup Environment

```zsh
git clone git@github.com:function2-llx/UMeI.git --recursive
cd UMeI
conda env create -n umei
conda activate umei
wandb login
```

## Run Stoic2021 Training
```zsh
cd umei/datasets/stoic2021
# place the original reference file at `origin/metadata/reference.csv`
python collect_clinical.py
cd ../../..
python train_<task>.py conf/<conf yaml file> [extra arguments...]
```
