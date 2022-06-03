# UMeI

U-shaped network for Medical Imaging

## Setup Environment

```zsh
git clone git@github.com:function2-llx/UMeI.git --recursive
cd UMeI
conda env create -n umei
conda activate umei
echo "PYTHONPATH=`pwd`" > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
# we use wandb as default logger
wandb login
```

## Run Experiments
```zsh
export CUDA_VISIBLE_DEVICES=...
python scripts/<task>/train.py <conf yaml file> [extra arguments...]
```

## Stoic2021

```zsh
cd umei/datasets/stoic2021
# place input data under this path; 
# also place the original reference file at `origin/metadata/reference.csv`
python collect_clinical.py  # generate clinical information, this should be run once 
cd ../../..
```

Download pre-trained model (you may use [gdown](https://github.com/wkentaro/gdown)): [Med3D](https://github.com/Tencent/MedicalNet)