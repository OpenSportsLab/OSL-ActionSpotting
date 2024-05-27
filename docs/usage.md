# Usage

## Configurations

The currently avaliable configs are the following:

Learneable Pooling:

```bash
configs/learnablepooling/json_netvlad++_resnetpca512.py
configs/learnablepooling/json_soccernet_netvlad++_resnetpca512.py

configs/learnablepooling/soccernet_avgpool_resnetpca512.py
configs/learnablepooling/soccernet_maxpool_resnetpca512.py
configs/learnablepooling/soccernet_netrvlad_resnetpca512.py
configs/learnablepooling/soccernet_netvlad_resnetpca512.py
configs/learnablepooling/soccernet_avgpool++_resnetpca512.py
configs/learnablepooling/soccernet_maxpool++_resnetpca512.py
configs/learnablepooling/soccernet_netrvlad++_resnetpca512.py
configs/learnablepooling/soccernet_netvlad++_resnetpca512.py
```

CALF:

```bash
configs/contextawarelossfunction/json_soccernet_calf_resnetpca512.py

configs/contextawarelossfunction/soccernet_resnetpca512.py
```

E2E-Spot:

```bash
configs/e2espot/e2espot.py
```

## Training

```bash
python tools/train.py {config}
```

### Train example

```bash
python tools/train.py \
    configs/learnablepooling/json_netvlad++_resnetpca512.py
```

### Train example with custom config file

```bash
python tools/train.py \
    configs/learnablepooling/soccernet_netvlad++_resnetpca512.py \
    --cfg-options training.max_epochs=10 \
        training.evaluation_frequency=8 \
        data_root=/datasets/SoccerNet \
        dataset.train.data_root=/datasets/SoccerNet \
        dataset.test.data_root=/datasets/SoccerNet \
        dataset.valid.data_root=/datasets/SoccerNet
```

## Inference and Evaluation

```bash
python tools/evaluate.py {config}
```

### Evaluation example

```bash
python tools/evaluate.py \
    configs/learnablepooling/json_netvlad++_resnetpca512.py
```

### Evaluation example with custom config file

```bash
python tools/evaluate.py \
    configs/learnablepooling/soccernet_netvlad++_resnetpca512.py \
    --cfg-options model.load_weights=/path/to/model.pth.tar \
        data_root=/datasets/SoccerNet \
        dataset.train.data_root=/datasets/SoccerNet \
        dataset.test.data_root=/datasets/SoccerNet \
        dataset.valid.data_root=/datasets/SoccerNet
```
