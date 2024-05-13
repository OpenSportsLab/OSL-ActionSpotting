import os
import logging
from datetime import datetime
import time
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch


import json


from oslspotting.core.utils.dali import get_repartition_gpu
from oslspotting.core.utils.eval import evaluate_e2e
from oslspotting.core.utils.lightning import CustomProgressBar
from oslspotting.datasets.builder import build_dataloader
from oslspotting.models import build_model
import pytorch_lightning as pl

from oslspotting.core.runner import infer_features, infer_game, infer_video


from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

def build_inferer(cfg, model):
    """Build a inferer from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        default_args (dict | None, optional): Default initialization arguments.
            Default: None.

    Returns:
        inferer: The constructed inferer.
    """
    if cfg.runner.type == "runner_e2e":
        inferer = Inferer(cfg=cfg, model=model)
    else :
        inferer = Inferer(cfg=cfg, model=model)
    return inferer

class Inferer():
    def __init__(self, cfg, model):
        
        self.cfg = cfg
        self.dali = getattr(cfg, 'dali', False)
        self.infer_split = getattr(cfg, 'infer_split', True)
        self.model = model

    def infer(self, data):
        if self.cfg.runner.type == 'runner_e2e':
            pred_file = None 
            if self.cfg.work_dir is not None:
                pred_file = os.path.join(self.cfg.work_dir, self.cfg.dataset.test.results)
                json_data = evaluate_e2e(self.model, getattr(self.cfg, 'dali', False), data, 'infer', self.cfg.classes, pred_file, False, True, self.cfg.dataset.test.dataloader, True)
                return json_data
        else:                  
            # Run Inference on Dataset
            infer_loader = build_dataloader(data, self.cfg.dataset.test.dataloader,self.cfg.training.GPU, self.dali)
            evaluator = pl.Trainer(callbacks=[CustomProgressBar()],devices=[self.cfg.training.GPU],num_sanity_val_steps=0)
            evaluator.predict(self.model,infer_loader)
            return self.model.json_data
        

# def init_spotter(
#     config: Union[str, Path, Config],
#     checkpoint: Optional[str] = None,
#     device: str = 'cuda:0',
#     cfg_options: Optional[dict] = None,
# ) -> nn.Module:
#     """Initialize a spotter from config file.

#     Args:
#         config (str, :obj:`Path`, or :obj:`mmengine.Config`): Config file path,
#             :obj:`Path`, or the config object.
#         checkpoint (str, optional): Checkpoint path. If left as None, the model
#             will not load any weights.
#         device (str): The device where the anchors will be put on.
#             Defaults to cuda:0.
#         cfg_options (dict, optional): Options to override some settings in
#             the used config.

#     Returns:
#         nn.Module: The constructed spotter.
#     """
#     if isinstance(config, (str, Path)):
#         config = Config.fromfile(config)
#     elif not isinstance(config, Config):
#         raise TypeError('config must be a filename or Config object, '
#                         f'but got {type(config)}')
#     if cfg_options is not None:
#         config.merge_from_dict(cfg_options)
#     elif 'init_cfg' in config.model.backbone:
#         config.model.backbone.init_cfg = None

#     # load checkpoint 
#     config.model.load_weights = checkpoint

#     # build model
#     model = build_model(config.model)
#     model.to(device)
#     model.eval()
#     return model



# def inference_spotter(
#     cfg: Union[str, Path, Config],
#     model: nn.Module,
#     input: str,
#     confidence_threshold: Optional[float] = 0.5,
# ):
#     """Inference image(s) with the detector.

#     Args:
#         model (nn.Module): The loaded detector.
#         imgs (str, ndarray, Sequence[str/ndarray]):
#            Either image files or loaded images.
#         test_pipeline (:obj:`Compose`): Test pipeline.

#     Returns:
#         :obj:`DetDataSample` or list[:obj:`DetDataSample`]:
#         If imgs is a list or tuple, the same length list type results
#         will be returned, otherwise return the detection results directly.
#     """
#     if cfg.runner.type == 'runner_e2e':
#         pred_file = None 
#         if cfg.work_dir is not None:
#             pred_file = os.path.join(cfg.work_dir, 'evaluate-{}'.format('infer'))
#             json_data= evaluate_e2e(model, getattr(cfg, 'dali', False), input, 'infer', cfg.classes, pred_file, False, True, cfg.dataset.test.dataloader, True)
#             return json_data
#     else:                  
#         evaluator = pl.Trainer(callbacks=[CustomProgressBar()],devices=[0],num_sanity_val_steps=0)
#         evaluator.predict(model,input)
#         return model.json_data
#     # if input.endswith(".npy"):
#     #     # Run Inference on Dataset
#     #     evaluator = pl.Trainer(callbacks=[CustomProgressBar()],devices=[0],num_sanity_val_steps=0)
#     #     evaluator.predict(model,input)
#     #     results = model.target_dir
#     #     json_results = infer_features(config, input, model, 
#     #                 confidence_threshold=confidence_threshold)
#     #     return json_results
#     # if os.path.isdir(input):
#     #     json_results = infer_game(config, input, model, 
#     #                 confidence_threshold=confidence_threshold)
#     #     return json_results
    