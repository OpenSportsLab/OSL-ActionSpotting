# How to setup a SLURM environment?

## Create conda environment

```bash
module load cuda/12.2
conda create -y -n osl python=3.11
conda activate osl
conda install pytorch==2.1.1 torchvision==0.16.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install --extra-index-url https://pypi.nvidia.com/ nvidia-dali-cuda120
pip install cupy-cuda12x
# pip install -r requirements.txt
# pip install "numpy<2.0"
pip install -e .
```

## Download dataset from HuggingFace



## Debug session

```bash
# Allocate ressources
srun --pty --time=2:00:00 --mem=40GB --gres=gpu:gtx1080ti:1 --ntasks=1 --cpus-per-task=4 bash -l

# Load module and setup environement
module load cuda/12.2
# module load gcc/6.4.0
module list
source activate osl
```

## SRUN for debugging sessions

```bash
# Allocate ressources
srun --pty --time=2:00:00 --mem=40GB --gres=gpu:gtx1080ti:1 --ntasks=1 --cpus-per-task=4 bash -l

# Load module and setup environement
module load cuda/12.2
# module load gcc/6.4.0
module list
source activate osl

# Run training code of any model
python tools/train.py configs/learnablepooling/json_netvlad++_resnetpca512.py \
--cfg-options \
dataset.train.path=/ibex/scratch/giancos/SoccerNet/spotting-OSL/ResNET_PCA512/train/annotations.json \
dataset.valid.path=/ibex/scratch/giancos/SoccerNet/spotting-OSL/ResNET_PCA512/valid/annotations.json \
dataset.test.path=/ibex/scratch/giancos/SoccerNet/spotting-OSL/ResNET_PCA512/test/annotations.json \
dataset.train.data_root=/ibex/scratch/giancos/SoccerNet/spotting-OSL/ResNET_PCA512/train \
dataset.valid.data_root=/ibex/scratch/giancos/SoccerNet/spotting-OSL/ResNET_PCA512/valid \
dataset.test.data_root=/ibex/scratch/giancos/SoccerNet/spotting-OSL/ResNET_PCA512/test


# Run evaluation code of any model
python tools/evaluate.py outputs/learnablepooling/json_netvlad++_resnetpca512/config.py \
--cfg-options \
model.load_weights=outputs/learnablepooling/json_netvlad++_resnetpca512/model.pth.tar
```

## Benchmark models on IBEX

```bash
sh tools/slurm/benchmark_on_ibex.sh 
```

This will run many jobs on IBEX that should each train specific models.
