# Tools for SN-Spotting

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
python tools/train.py configs/learnablepooling/soccernet_netvlad++_resnetpca512.py --cfg-options training.max_epochs=10 training.evaluation_frequency=8
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
python tools/evaluate.py configs/learnablepooling/soccernet_netvlad++_resnetpca512.py --cfg-options model.load_weights=/home/giancos/git/sn-spotting/Benchmarks/TemporallyAwarePooling/models/NetVLAD++/model.pth.tar
```

## Infer

```bash
python tools/infer.py \
configs/learnablepooling/soccernet_netvlad++_resnetpca512.py \
--input "/home/giancos/git/sn-spotting-pip/path/to/SoccerNet/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/1_ResNET_TF2_PCA512.npy" \
--checkpoint outputs/learnablepooling/soccernet_netvlad++_resnetpca512/model.pth.tar \
--overwrite


python tools/infer.py \
configs/learnablepooling/soccernet_netvlad++_resnetpca512.py \
--input "/home/giancos/git/sn-spotting-pip/path/to/SoccerNet/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley" \
--checkpoint outputs/learnablepooling/soccernet_netvlad++_resnetpca512/model.pth.tar \
--overwrite
```

## Extract feratures

```bash
pip install scikit-video tensorflow==2.3.0 imutils opencv-python==3.4.11.41 SoccerNet moviepy scikit-learn ffmpy protobuf==3.20 ffmpeg ffmpy
```

```bash
python tools/features/extract_features.py --path_video path/to/video.mkv --path_features path/to/features.npy -PCA tools/features/pca_512_TF2.pkl --PCA_scaler tools/features/average_512_TF2.pkl
```
