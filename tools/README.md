# Tools

## Download pre-trained weights

### From server

Visit `https://exrcsdrive.kaust.edu.sa/index.php/s/eIjTapzHicsb4yy` (pw:OSL) and fetch the config and weights files for your model of interest.

### From python script

Directly fetch the files (config or weights) for your model of interest.

```bash
python tools/download_weights_model_zoo.py --Model avgpool --File json_avgpool_resnetpca512.py
python tools/download_weights_model_zoo.py --Model avgpool --File model.pth.tar
```
