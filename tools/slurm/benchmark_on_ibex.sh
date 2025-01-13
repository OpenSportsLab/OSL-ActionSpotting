# Download data

## Activate the conda environment
conda activate osl
pip install huggingface_hub[cli]

# Download SoccerNet Ball Action Spotting - 224p videos
python tools/download_dataset_huggingface.py \
--dataset_repo=OpenSportsLab/SoccerNet-BallActionSpotting-Videos \
--output_dir=/ibex/scratch/giancos/OSL/HuggingFace/SoccerNet-BallActionSpotting-Videos \
--allow_patterns="224p/*"

# Download SoccerNet Action Spotting - 224p videos
python tools/download_dataset_huggingface.py \
--dataset_repo=OpenSportsLab/SoccerNet-ActionSpotting-Videos \
--output_dir=/ibex/scratch/giancos/OSL/HuggingFace/SoccerNet-ActionSpotting-Videos \
--allow_patterns="224p/*"

# Download SoccerNet Action Spotting - ResNET512
python tools/download_dataset_huggingface.py \
--dataset_repo=OpenSportsLab/SoccerNet-ActionSpotting-Features \
--output_dir=/ibex/scratch/giancos/OSL/HuggingFace/SoccerNet-ActionSpotting-Features \
--allow_patterns="ResNET_PCA512/*"



# # Run SLURM batch jobs

# ## NetVLAD++ on SoccerNet Action Spotting dataset (ResNet PCA512)
# sbatch --job-name=json_netvlad++_resnetpca512 --time=2:00:00 --mem=40GB --gres=gpu:gtx1080ti:1 --ntasks=1 --cpus-per-task=4 \
# tools/slurm/train.sh \
#  configs/learnablepooling/json_netvlad++_resnetpca512.py \
# --cfg-options \
# dataset.train.path=/ibex/scratch/giancos/SoccerNet/spotting-OSL/ResNET_PCA512/train/annotations.json \
# dataset.valid.path=/ibex/scratch/giancos/SoccerNet/spotting-OSL/ResNET_PCA512/valid/annotations.json \
# dataset.test.path=/ibex/scratch/giancos/SoccerNet/spotting-OSL/ResNET_PCA512/test/annotations.json \
# dataset.train.data_root=/ibex/scratch/giancos/SoccerNet/spotting-OSL/ResNET_PCA512/train \
# dataset.valid.data_root=/ibex/scratch/giancos/SoccerNet/spotting-OSL/ResNET_PCA512/valid \
# dataset.test.data_root=/ibex/scratch/giancos/SoccerNet/spotting-OSL/ResNET_PCA512/test


# sbatch --job-name=e2espot --time=2-00:00:00 --mem=200GB --gres=gpu:v100:4 --ntasks=1 --cpus-per-task=16 \
# tools/slurm/train.sh \
#  configs/e2espot/e2espot.py \
# --cfg-options \
# dataset.train.path=/ibex/scratch/giancos/OSL/HuggingFace/SoccerNet-BallActionSpotting-Videos/224p/train/annotations-2024-224p.json \
# dataset.valid.path=/ibex/scratch/giancos/OSL/HuggingFace/SoccerNet-BallActionSpotting-Videos/224p/valid/annotations-2024-224p.json \
# dataset.valid_data_frames.path=/ibex/scratch/giancos/OSL/HuggingFace/SoccerNet-BallActionSpotting-Videos/224p/valid/annotations-2024-224p.json \
# dataset.test.path=/ibex/scratch/giancos/OSL/HuggingFace/SoccerNet-BallActionSpotting-Videos/224p/test/annotations-2024-224p.json \
# dataset.train.data_root=/ibex/scratch/giancos/OSL/HuggingFace/SoccerNet-BallActionSpotting-Videos/224p/train \
# dataset.valid.data_root=/ibex/scratch/giancos/OSL/HuggingFace/SoccerNet-BallActionSpotting-Videos/224p/valid \
# dataset.valid_data_frames.data_root=/ibex/scratch/giancos/OSL/HuggingFace/SoccerNet-BallActionSpotting-Videos/224p/valid \
# dataset.test.data_root=/ibex/scratch/giancos/OSL/HuggingFace/SoccerNet-BallActionSpotting-Videos/224p/test \
# model.multi_gpu=True \
# training.GPU=4


# sbatch --job-name=e2espot --time=4-00:00:00 --mem=90GB --gres=gpu:a100:1 --ntasks=1 --cpus-per-task=6 \
# tools/slurm/train.sh \
#  configs/e2espot/e2espot.py \
# --cfg-options \
# dataset.train.path=/ibex/scratch/giancos/OSL/HuggingFace/SoccerNet-BallActionSpotting-Videos/224p/train/annotations-2024-224p.json \
# dataset.valid.path=/ibex/scratch/giancos/OSL/HuggingFace/SoccerNet-BallActionSpotting-Videos/224p/valid/annotations-2024-224p.json \
# dataset.valid_data_frames.path=/ibex/scratch/giancos/OSL/HuggingFace/SoccerNet-BallActionSpotting-Videos/224p/valid/annotations-2024-224p.json \
# dataset.test.path=/ibex/scratch/giancos/OSL/HuggingFace/SoccerNet-BallActionSpotting-Videos/224p/test/annotations-2024-224p.json \
# dataset.train.data_root=/ibex/scratch/giancos/OSL/HuggingFace/SoccerNet-BallActionSpotting-Videos/224p/train \
# dataset.valid.data_root=/ibex/scratch/giancos/OSL/HuggingFace/SoccerNet-BallActionSpotting-Videos/224p/valid \
# dataset.valid_data_frames.data_root=/ibex/scratch/giancos/OSL/HuggingFace/SoccerNet-BallActionSpotting-Videos/224p/valid \
# dataset.test.data_root=/ibex/scratch/giancos/OSL/HuggingFace/SoccerNet-BallActionSpotting-Videos/224p/test


# # E2E on Action Spotting dataset (224p) on 4 GPU v100
# sbatch --job-name=e2espot --time=4-00:00:00 --mem=200GB --gres=gpu:v100:4 --ntasks=1 --cpus-per-task=16 \
# tools/slurm/train.sh \
#  configs/e2espot/e2espot.py \
# --cfg-options \
# dataset.train.path=/ibex/scratch/giancos/OSL/HuggingFace/SoccerNet-ActionSpotting-Videos/224p/train/annotations.json \
# dataset.valid.path=/ibex/scratch/giancos/OSL/HuggingFace/SoccerNet-ActionSpotting-Videos/224p/valid/annotations.json \
# dataset.valid_data_frames.path=/ibex/scratch/giancos/OSL/HuggingFace/SoccerNet-ActionSpotting-Videos/224p/valid/annotations.json \
# dataset.test.path=/ibex/scratch/giancos/OSL/HuggingFace/SoccerNet-ActionSpotting-Videos/224p/test/annotations.json \
# dataset.train.data_root=/ibex/scratch/giancos/OSL/HuggingFace/SoccerNet-ActionSpotting-Videos/224p/train \
# dataset.valid.data_root=/ibex/scratch/giancos/OSL/HuggingFace/SoccerNet-ActionSpotting-Videos/224p/valid \
# dataset.valid_data_frames.data_root=/ibex/scratch/giancos/OSL/HuggingFace/SoccerNet-ActionSpotting-Videos/224p/valid \
# dataset.test.data_root=/ibex/scratch/giancos/OSL/HuggingFace/SoccerNet-ActionSpotting-Videos/224p/test \
# model.multi_gpu=True \
# training.GPU=4
