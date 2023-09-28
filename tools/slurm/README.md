# How to setup a SLURM environment?

## Create environment

```bash
conda create -y -n snspotting python=3.8
conda activate snspotting
conda install -y pytorch=1.6 torchvision=0.7 cudatoolkit=10.1 -c pytorch
pip install SoccerNet matplotlib scikit-learn sklearn
pip install mmengine
pip install -e .
conda deactivate
```

## Debug session

```bash
# Allocate ressources
srun --pty --time=4:00:00 --gres=gpu:gtx1080ti:1 bash -l

# Load module and setup environement
module load cuda/10.2.89
module load gcc/6.4.0
module list
source activate snspotting
```

## Link datasets from scratch

Create symbolic link for more storage:

```bash
mkdir -p path/to/
ln -s /ibex/scratch/giancos/SoccerNet path/to/SoccerNet
```

## SRUN for debugging sessions

```bash
# Allocate ressources
srun --pty --time=4:00:00 --gres=gpu:gtx1080ti:1 bash -l

# Load module and setup environement
module load cuda/10.2.89
module load gcc/6.4.0
module list
source activate snspotting

# Run training code of any code
python tools/train.py configs/learnablepooling/soccernet_netvlad++_resnetpca512.py
```

## SBATCH to run experiments

Run training and evaluation python code:

```bash
# create log folder
mkdir -p log

# Temporally Aware Learnable Pooling
sbatch -J SN_NV+_RN512  --time 04:00:00 tools/slurm/slurm_train_eval.sh configs/learnablepooling/soccernet_netvlad++_resnetpca512.py
sbatch -J SN_NV_RN512   --time 04:00:00 tools/slurm/slurm_train_eval.sh configs/learnablepooling/soccernet_netvlad_resnetpca512.py
sbatch -J SN_NRV+_RN512 --time 04:00:00 tools/slurm/slurm_train_eval.sh configs/learnablepooling/soccernet_netrvlad++_resnetpca512.py
sbatch -J SN_NRV_RN512  --time 04:00:00 tools/slurm/slurm_train_eval.sh configs/learnablepooling/soccernet_netrvlad_resnetpca512.py
sbatch -J SN_MP+_RN512  --time 04:00:00 tools/slurm/slurm_train_eval.sh configs/learnablepooling/soccernet_maxpool++_resnetpca512.py
sbatch -J SN_MP_RN512   --time 04:00:00 tools/slurm/slurm_train_eval.sh configs/learnablepooling/soccernet_maxpool_resnetpca512.py
sbatch -J SN_AP+_RN512  --time 04:00:00 tools/slurm/slurm_train_eval.sh configs/learnablepooling/soccernet_avgpool++_resnetpca512.py
sbatch -J SN_AP_RN512   --time 04:00:00 tools/slurm/slurm_train_eval.sh configs/learnablepooling/soccernet_avgpool_resnetpca512.py

# Context Aware Loss Function
sbatch -J SN_CALF_RN512   --time 24:00:00 tools/slurm/slurm_train_eval.sh configs/contextawarelossfunction/soccernet_resnetpca512.py 
```
