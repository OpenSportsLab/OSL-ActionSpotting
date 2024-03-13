#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=spot            # Name of the job
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --cpus-per-task=4           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=4G            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:4               # Number of GPU's
#SBATCH --time=120:00:00              # Max execution time
#SBATCH --partition=a5000

# Activate your Anaconda environment
source /home/ybenzakour/anaconda3/etc/profile.d/conda.sh
conda activate snspotting # CHANGEME
git checkout yassine
# Run your Python script
nvidia-smi
python tools/train.py configs/e2espot/e2espot.py --cfg-options num_epochs=100 dataset.crop_dim=-1
# python tools/train_e2e.py soccernetv2 /home/ybenzakour/frames -s rny002_gsm -m rny002_gsm --num_epochs 100 --crop_dim -1 --mixup --print_gpus --epoch_num_frames 500000 --dali