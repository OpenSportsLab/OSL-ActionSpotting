# sn-spotting-pip

```bash
conda create -y -n snspotting python=3.9
conda activate snspotting
conda install -y pytorch torchvision -c pytorch -c nvidia
conda install -y cudatoolkit=11.8 -c pytorch
pip install SoccerNet matplotlib scikit-learn sklearn cupy pytorch-lightning opencv-python moviepy tqdm
pip install mmengine
pip install -e .
```

## TODO

[x] Create pip package setup
[x] Push library to pypi
[] Integrate CALF -> Need fix for parameters K forced to cuda!
[] Integrate PTS
[] Integrate Pytorch Lightning

