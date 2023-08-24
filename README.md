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
python tools/train.py configs/netvlad++_resnet512_soccernet.py --cfg-options data_root=/media/giancos/Football/SoccerNet/
```
