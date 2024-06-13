# How to setup a SLURM environment?

## Debug session

```bash
# Allocate ressources
srun --pty --time=4:00:00 --gres=gpu:gtx1080ti:1 bash -l

# Load module and setup environement
module load cuda/10.2.89
module load gcc/6.4.0
module list
source activate oslactionspotting
```
## SRUN for debugging sessions

```bash
# Allocate ressources
srun --pty --time=4:00:00 --gres=gpu:gtx1080ti:1 bash -l

# Load module and setup environement
module load cuda/10.2.89
module load gcc/6.4.0
module list
source activate oslactionspotting

# Run training code of any code
python tools/train.py configs/learnablepooling/soccernet_netvlad++_resnetpca512.py
```