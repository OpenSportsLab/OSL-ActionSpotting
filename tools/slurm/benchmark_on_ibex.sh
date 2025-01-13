# Download data

source activate osl
pip install huggingface_hub[cli]
python tools/download_dataset_huggingface.py \
--dataset_repo=SoccerNet/SN-BAS-2025 --output_dir=/ibex/scratch/giancos/SoccerNet/HuggingFace/SN-BAS-2025
python tools/download_dataset_huggingface.py \
--dataset_repo=SoccerNet/SN-BAS-2024 --output_dir=/ibex/scratch/giancos/SoccerNet/HuggingFace/SN-BAS-2024


# Run SLURM batch jobs

## NetVLAD++ on SoccerNet Action Spotting dataset (ResNet PCA512)
sbatch --job-name=json_netvlad++_resnetpca512 --time=2:00:00 --mem=40GB --gres=gpu:gtx1080ti:1 --ntasks=1 --cpus-per-task=4 \
tools/slurm/train.sh \
 configs/learnablepooling/json_netvlad++_resnetpca512.py \
--cfg-options \
dataset.train.path=/ibex/scratch/giancos/SoccerNet/spotting-OSL/ResNET_PCA512/train/annotations.json \
dataset.valid.path=/ibex/scratch/giancos/SoccerNet/spotting-OSL/ResNET_PCA512/valid/annotations.json \
dataset.test.path=/ibex/scratch/giancos/SoccerNet/spotting-OSL/ResNET_PCA512/test/annotations.json \
dataset.train.data_root=/ibex/scratch/giancos/SoccerNet/spotting-OSL/ResNET_PCA512/train \
dataset.valid.data_root=/ibex/scratch/giancos/SoccerNet/spotting-OSL/ResNET_PCA512/valid \
dataset.test.data_root=/ibex/scratch/giancos/SoccerNet/spotting-OSL/ResNET_PCA512/test

# sbatch --job-name=e2espot --time=24:00:00 --mem=90GB --gres=gpu:v100:1 --ntasks=1 --cpus-per-task=6 \
# tools/slurm/train.sh \
#  configs/e2espot/e2espot.py \
# --cfg-options \
# dataset.train.path=/ibex/scratch/giancos/SoccerNet/spotting-OSL/224p/train/annotations.json \
# dataset.valid.path=/ibex/scratch/giancos/SoccerNet/spotting-OSL/224p/valid/annotations.json \
# dataset.test.path=/ibex/scratch/giancos/SoccerNet/spotting-OSL/224p/test/annotations.json \
# dataset.train.data_root=/ibex/scratch/giancos/SoccerNet/spotting-OSL/224p/train \
# dataset.valid.data_root=/ibex/scratch/giancos/SoccerNet/spotting-OSL/224p/valid \
# dataset.test.data_root=/ibex/scratch/giancos/SoccerNet/spotting-OSL/224p/test

