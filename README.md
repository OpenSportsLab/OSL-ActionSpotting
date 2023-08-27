# sn-spotting-pip

```bash
conda create -y -n snspotting python=3.8
conda activate snspotting
conda install -y pytorch=1.6 torchvision=0.7 cudatoolkit=10.1 -c pytorch
pip install SoccerNet matplotlib sklearn
pip install mmengine
```



## Tools 

## Training

```bash
python tools/train.py configs/soccernet_learnablepooling_maxpool_resnetpca512.py
python tools/train.py configs/soccernet_learnablepooling_avgpool_resnetpca512.py
python tools/train.py configs/soccernet_learnablepooling_netrvlad_resnetpca512.py
python tools/train.py configs/soccernet_learnablepooling_netvlad_resnetpca512.py

python tools/train.py configs/soccernet_learnablepooling_maxpool++_resnetpca512.py
python tools/train.py configs/soccernet_learnablepooling_avgpool++_resnetpca512.py
python tools/train.py configs/soccernet_learnablepooling_netrvlad++_resnetpca512.py
python tools/train.py configs/soccernet_learnablepooling_netvlad++_resnetpca512.py
```
