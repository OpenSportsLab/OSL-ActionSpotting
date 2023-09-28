#!/bin/bash
#SBATCH -J snspot
#SBATCH -o log/%x.%3a.%A.out
#SBATCH -e log/%x.%3a.%A.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:gtx1080ti:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH -N 1


date

echo "Loading anaconda..."
# module purge
# module load anaconda3
module load cuda/10.2.89
module load gcc/6.4.0
module list
source activate snspotting
echo "...Anaconda env loaded"


echo "Running Training python script..."
python tools/train.py $1
echo "... training done."

echo "Running Testing python script..."
python tools/evaluate.py $1
echo "... testing done."

date
