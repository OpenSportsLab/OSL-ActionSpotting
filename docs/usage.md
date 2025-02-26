# Usage

## Configurations

The currently avaliable configs are the following:

Learneable Pooling:

```bash
configs/learnablepooling/soccernet_netvlad++_resnetpca512.py

configs/learnablepooling/json_avgpool_resnetpca512.py
configs/learnablepooling/json_maxpool_resnetpca512.py
configs/learnablepooling/json_netrvlad_resnetpca512.py
configs/learnablepooling/json_netvlad_resnetpca512.py
configs/learnablepooling/json_avgpool++_resnetpca512.py
configs/learnablepooling/json_maxpool++_resnetpca512.py
configs/learnablepooling/json_netrvlad++_resnetpca512.py
configs/learnablepooling/json_netvlad++_resnetpca512.py
```

CALF:

```bash
configs/contextawarelossfunction/json_soccernet_calf_resnetpca512.py

configs/contextawarelossfunction/soccernet_resnetpca512.py
```

PTS:

```bash
configs/e2espot/e2espot.py
configs/e2espot/e2espot_ocv.py
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
    configs/learnablepooling/json_netvlad++_resnetpca512.py \
    --cfg-options training.max_epochs=10 \
        dataset.train.data_root=/datasets/SoccerNet \
        dataset.valid.data_root=/datasets/SoccerNet \
        dataset.train.path=/datasets/SoccerNet/ResNET_PCA512/train/annotations.json \
        dataset.valid.path=/datasets/SoccerNet/ResNET_PCA512/valid/annotations.json
```

## Inference

```bash
python tools/infer.py {config}
```

### Inference example

```bash
python tools/infer.py \
    configs/learnablepooling/json_netvlad++_resnetpca512.py
```

#### For E2E, provide the model weights path

```bash
python tools/infer.py \
    configs/e2espot/e2espot.py --weights /path/to/your/model/weights
```

_Note:- If you don't provide the path to the model weights, the weights are assumed to be inside the cfg.work_dir as "best_checkpoint.pt"_

### Inference example with custom config file

```bash
python tools/infer.py \
    configs/learnablepooling/json_netvlad++_resnetpca512.py \
    --cfg-options dataset.test.data_root=/datasets/SoccerNet \
        dataset.test.path=/datasets/SoccerNet/ResNET_PCA512/test/annotations.json
```

## Evaluation

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
    configs/learnablepooling/json_netvlad++_resnetpca512.py \
    --cfg-options dataset.test.path=/datasets/SoccerNet/ResNET_PCA512/test/annotations.json \
    dataset.test.metric=tight
```

### If you don't see any results, you might need to specify the path to your results file

```bash
python tools/evaluate.py \
    configs/e2espot/e2espot.py \
    --cfg-options dataset.test.results=/outputs/e2e/rny008_gsm_150/results_spotting_test.recall.json.gz
```

*Try to provide full path to the results, if you do not see any results.*
#### So for a custom evaluation on test set, it might look something like this

```bash
python tools/evaluate.py \
    configs/e2espot/e2espot.py \
    --cfg-options dataset.test.path=/datasets/224p/test/annotations.json \
    dataset.test.data_root=/datasets/224p/test \
    dataset.test.results=outputs/e2e/rny008_gsm_150/results_spotting_test.recall.json.gz \
    dataset.test.metric=tight
```
## Visualization

```bash
python tools/visualize.py {config}
```

### Visualization example

```bash
python tools/visualize.py \
    configs/learnablepooling/json_netvlad++_resnetpca512.py
```

### Visualization example with custom config file

```bash
python tools/visualize.py \
    configs/learnablepooling/json_netvlad++_resnetpca512.py \
    --cfg-options dataset.test.results=/outputs/learnablepooling/json_netvlad++_resnetpca512/results_spotting_test/england_epl/2014-2015/2015-05-17_-_18-00_Manchester_United_1_-_1_Arsenal/1_ResNET_TF2_PCA512/results_spotting.json \
    dataset.test.path=/home/ybenzakour/datasets/SoccerNet/england_epl/2014-2015/2015-05-17_-_18-00_Manchester_United_1_-_1_Arsenal/1_224p.mkv \
    visualizer.threshold=0.2
```

