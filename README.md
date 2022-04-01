# UMeI

U-shaped network for Medical Imaging

## Setup Environment

```zsh
git clone git@github.com:function2-llx/UMeI.git --recursive
cd UMeI
conda env create -n umei
```

## Run
```zsh
conda activate umei
wandb login
python main.py conf/<name>.yml [args...]
```
