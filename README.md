# sn-spotting-pip

```bash
conda create -y -n snspotting python=3.8
conda activate snspotting
conda install -y pytorch=1.6 torchvision=0.7 cudatoolkit=10.1 -c pytorch
pip install SoccerNet matplotlib scikit-learn sklearn
pip install mmengine
pip install -e .
```

## TODO

[x] Create pip package setup
[x] Push library to pypi
[] Integrate CALF -> Need fix for parameters K forced to cuda!
[] Integrate PTS
[] Integrate Pytorch Lightning

