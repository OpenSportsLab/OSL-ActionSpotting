

import os
import logging
from datetime import datetime
import time
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
import mmengine
from mmengine.config import Config, DictAction


import json


from snspotting.datasets import build_dataset, build_dataloader
from snspotting.models import build_model
from snspotting.loss import build_criterion
from snspotting.core import build_optimizer, build_scheduler

from snspotting.core.inference import infer_features, infer_game, infer_video
from snspotting.core.evaluation import evaluate_Spotting #testClassication, testSpotting

import copy
import warnings
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn


def init_spotter(
    config: Union[str, Path, Config],
    checkpoint: Optional[str] = None,
    device: str = 'cuda:0',
    cfg_options: Optional[dict] = None,
) -> nn.Module:
    """Initialize a spotter from config file.

    Args:
        config (str, :obj:`Path`, or :obj:`mmengine.Config`): Config file path,
            :obj:`Path`, or the config object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str): The device where the anchors will be put on.
            Defaults to cuda:0.
        cfg_options (dict, optional): Options to override some settings in
            the used config.

    Returns:
        nn.Module: The constructed spotter.
    """
    if isinstance(config, (str, Path)):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    elif 'init_cfg' in config.model.backbone:
        config.model.backbone.init_cfg = None

    # load checkpoint 
    config.model.load_weights = checkpoint

    # build model
    model = build_model(config.model)
    model.to(device)
    model.eval()
    return model



def inference_spotter(
    config: Union[str, Path, Config],
    model: nn.Module,
    input: str,
    # test_pipeline: Optional[Compose] = None,
    confidence_threshold: Optional[float] = 0.5,
    text_prompt: Optional[str] = None,
    custom_entities: bool = False,
) -> dict: # Union[DetDataSample, SampleList]:
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str, ndarray, Sequence[str/ndarray]):
           Either image files or loaded images.
        test_pipeline (:obj:`Compose`): Test pipeline.

    Returns:
        :obj:`DetDataSample` or list[:obj:`DetDataSample`]:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """
    if isinstance(config, (str, Path)):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
                        

    if input.endswith(".npy"):
        json_results = infer_features(config, input, model, 
                    confidence_threshold=confidence_threshold)
        return json_results
    if os.path.isdir(input):
        json_results = infer_game(config, input, model, 
                    confidence_threshold=confidence_threshold)
        return json_results
    