import torch
import os

from snspotting.core.utils import CustomProgressBar, MyCallback
import pytorch_lightning as pl

# from .training import * 
# from .evaluation import *

def build_trainer(cfg, default_args=None):
    call=MyCallback()
    trainer = pl.Trainer(max_epochs=cfg.max_epochs,devices=[cfg.GPU],callbacks=[call,CustomProgressBar(refresh_rate=1)],num_sanity_val_steps=0)
    return trainer