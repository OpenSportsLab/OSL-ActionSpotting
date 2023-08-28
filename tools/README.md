# Tools for SN-Spotting

## Training

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


### Overwrite the config file

```bash
python tools/train.py configs/learnablepooling/soccernet_netvlad++_resnetpca512.py --cfg-options training.max_epochs=10 training.evaluation_frequency=8
```


## Evaluate

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


## Infer

```bash
python tools/evaluate.py \
configs/soccernet_learnablepooling_netvlad++_resnetpca512.py \
--weights models/soccernet_learnablepooling_netvlad++_resnetpca512/model.pth.tar
--video path/to/video
```
