# sn-spotting-pip

```bash
conda create -y -n snspotting python=3.8
conda activate snspotting
conda install -y pytorch=1.6 torchvision=0.7 cudatoolkit=10.1 -c pytorch
pip install SoccerNet matplotlib scikit-learn sklearn
pip install mmengine
```



## Tools 

### Training

```bash
python tools/train.py configs/learnablepooling/soccernet_maxpool_resnetpca512.py
python tools/train.py configs/learnablepooling/soccernet_avgpool_resnetpca512.py
python tools/train.py configs/learnablepooling/soccernet_netrvlad_resnetpca512.py
python tools/train.py configs/learnablepooling/soccernet_netvlad_resnetpca512.py
python tools/train.py configs/learnablepooling/soccernet_maxpool++_resnetpca512.py
python tools/train.py configs/learnablepooling/soccernet_avgpool++_resnetpca512.py
python tools/train.py configs/learnablepooling/soccernet_netrvlad++_resnetpca512.py
python tools/train.py configs/learnablepooling/soccernet_netvlad++_resnetpca512.py
```


#### Overwrite the config file

```bash
python tools/train.py configs/soccernet_learnablepooling_netvlad++_resnetpca512.py --cfg-options training.max_epochs=10 training.evaluation_frequency=8
```


### Evaluate

```bash
python tools/evaluate.py configs/soccernet_learnablepooling_maxpool_resnetpca512.py
python tools/evaluate.py configs/soccernet_learnablepooling_avgpool_resnetpca512.py
python tools/evaluate.py configs/soccernet_learnablepooling_netrvlad_resnetpca512.py
python tools/evaluate.py configs/soccernet_learnablepooling_netvlad_resnetpca512.py

python tools/evaluate.py configs/soccernet_learnablepooling_maxpool++_resnetpca512.py
python tools/evaluate.py configs/soccernet_learnablepooling_avgpool++_resnetpca512.py
python tools/evaluate.py configs/soccernet_learnablepooling_netrvlad++_resnetpca512.py
python tools/evaluate.py configs/soccernet_learnablepooling_netvlad++_resnetpca512.py --weights models/soccernet_learnablepooling_netvlad++_resnetpca512/model.pth.tar
```
