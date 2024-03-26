# sn-spotting-pip

```bash
conda create -y -n snspotting python=3.9
conda activate snspotting
conda install -y pytorch torchvision -c pytorch -c nvidia
conda install -y cudatoolkit=11.8 -c pytorch
pip install SoccerNet matplotlib scikit-learn cupy-cuda11x pytorch-lightning opencv-python moviepy tqdm tabulate nvidia-dali-cuda110 timm
pip install mmengine
pip install -e .
```
# Tools for SN-Spotting for training and evaluating

## Training

```bash
python tools/train.py configs/learnablepooling/soccernet_avgpool_resnetpca512.py
python tools/train.py configs/learnablepooling/soccernet_maxpool_resnetpca512.py
python tools/train.py configs/learnablepooling/soccernet_netrvlad_resnetpca512.py
python tools/train.py configs/learnablepooling/soccernet_netvlad_resnetpca512.py
python tools/train.py configs/learnablepooling/soccernet_avgpool++_resnetpca512.py
python tools/train.py configs/learnablepooling/soccernet_maxpool++_resnetpca512.py
python tools/train.py configs/learnablepooling/soccernet_netrvlad++_resnetpca512.py
python tools/train.py configs/learnablepooling/soccernet_netvlad++_resnetpca512.py


python tools/train.py configs/contextawarelossfunction/soccernet_resnetpca512.py
```

### Train with custom config file

```bash
python tools/train.py configs/learnablepooling/soccernet_netvlad++_resnetpca512.py --cfg-options training.max_epochs=10 training.evaluation_frequency=8 data_root=/datasets/SoccerNet dataset.train.data_root=/datasets/SoccerNet dataset.test.data_root=/datasets/SoccerNet dataset.val.data_root=/datasets/SoccerNet
```

## Evaluate

```bash
python tools/evaluate.py configs/learnablepooling/soccernet_avgpool_resnetpca512.py
python tools/evaluate.py configs/learnablepooling/soccernet_maxpool_resnetpca512.py
python tools/evaluate.py configs/learnablepooling/soccernet_netrvlad_resnetpca512.py
python tools/evaluate.py configs/learnablepooling/soccernet_netvlad_resnetpca512.py
python tools/evaluate.py configs/learnablepooling/soccernet_avgpool++_resnetpca512.py
python tools/evaluate.py configs/learnablepooling/soccernet_maxpool++_resnetpca512.py
python tools/evaluate.py configs/learnablepooling/soccernet_netrvlad++_resnetpca512.py
python tools/evaluate.py configs/learnablepooling/soccernet_netvlad++_resnetpca512.py

python tools/evaluate.py configs/contextawarelossfunction/soccernet_resnetpca512.py

```

### Evaluate with custom config file

```bash
python tools/evaluate.py configs/learnablepooling/soccernet_netvlad++_resnetpca512.py --cfg-options model.load_weights=/home/giancos/git/sn-spotting/Benchmarks/TemporallyAwarePooling/models/NetVLAD++/model.pth.tar data_root=/datasets/SoccerNet dataset.train.data_root=/datasets/SoccerNet dataset.test.data_root=/datasets/SoccerNet dataset.val.data_root=/datasets/SoccerNet
```

## TODO

[x] Create pip package setup
[x] Push library to pypi
[X] Integrate CALF -> Need fix for parameters K forced to cuda!
[] Integrate PTS
[X] Integrate Pytorch Lightning

