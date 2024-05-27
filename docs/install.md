# Installation

## **Step 1** Create the environment by replacing x by the wanted python version : {3.9}, {3.10} or {3.11}.

```bash
conda create -y -n osl-action-spotting python=x
```

## **Step 2** Activate the environment

```bash
conda activate osl-action-spotting
```

## **Step 3** Install the library by replacing x, y and z depending on your cuda version.
## If cuda version is {11.2} ~ {11.x} : x, y and z are respectively {11.8}, {110} and {11x}.
## If cuda version is {12.x} : x, y and z are respectively {12.1}, {120} and {12x}.

```bash
conda install pytorch==2.1.1 torchvision==0.16.1 pytorch-cuda=x -c pytorch -c nvidia
conda install -y cudatoolkit=x -c pytorch
pip install nvidia-dali-cuday
pip install cupy-cudaz
pip install -r requirements.txt
pip install -e .
```
