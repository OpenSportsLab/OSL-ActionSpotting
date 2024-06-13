#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=spot            # Name of the job
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --cpus-per-task=2           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=4G            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:4               # Number of GPU's
#SBATCH --time=120:00:00              # Max execution time

# Activate your Anaconda environment
source /home/ybenzakour/anaconda3/etc/profile.d/conda.sh #REPLACE ME
conda activate osl-action-spotting # CHANGEME


python tools/train.py configs/e2espot/e2espot.py --cfg-options model.multi_gpu=True work_dir=outputs/e2e/rny002_gsm_trois model.backbone.type=rny002_gsm
# python tools/infer.py configs/e2espot/e2espot.py --cfg-options model.multi_gpu=True dataset.test.results=results_test_dali_2 work_dir=outputs/e2e/rny002_gsm_trois model.backbone.type=rny002_gsm
# python tools/infer.py configs/e2espot/e2espot.py --cfg-options model.multi_gpu=True dataset.test.results=results_test_dali_2
# python tools/infer.py configs/contextawarelossfunction/json_soccernet_calf_resnetpca512.py --cfg-options dataset.test.path=/home/ybenzakour/datasets/SoccerNet/england_epl/2014-2015/2015-05-17_-_18-00_Manchester_United_1_-_1_Arsenal/1_224p.mkv
# python tools/train.py configs/e2espot/e2espot.py --cfg-options training.num_epochs=150 model.feature_arch=rny008_gsm model.multi_gpu=True
# python tools/train_e2e.py soccernetv2 /home/ybenzakour/frames -s rny002_gsm -m rny002_gsm --num_epochs 100 --crop_dim -1 --mixup --print_gpus --epoch_num_frames 500000 --dali
# python tools/train.py configs/e2espot/e2espot.py --cfg-options work_dir=outputs/e2e/rny002_gsm_actionball classes=datasets_jsons/actionball/2_fps/dali/class.txt dataset.train.data_root=/home/ybenzakour/datasets/ActionBall/spotting-ball-2023/spotting-ball-2023 dataset.val.data_root=/home/ybenzakour/datasets/ActionBall/spotting-ball-2023/spotting-ball-2023 dataset.test.data_root=/home/ybenzakour/datasets/ActionBall/spotting-ball-2023/spotting-ball-2023 dataset.challenge.data_root=/home/ybenzakour/datasets/ActionBall/spotting-ball-2023/spotting-ball-2023 dataset.val_data_frames.data_root=/home/ybenzakour/datasets/ActionBall/spotting-ball-2023/spotting-ball-2023 dataset.epoch_num_frames=6700 dataset.train.label_file=datasets_jsons/actionball/2_fps/dali/train.json dataset.val.label_file=datasets_jsons/actionball/2_fps/dali/val.json dataset.val_data_frames.label_file=datasets_jsons/actionball/2_fps/dali/val.json dataset.test.label_file=datasets_jsons/actionball/2_fps/dali/test.json dataset.challenge.label_file=datasets_jsons/actionball/2_fps/dali/challenge.json