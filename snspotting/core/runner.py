import torch
import os

from .inference import *



def build_runner(cfg, model=None, default_args=None):
    """Build a runner from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        default_args (dict | None, optional): Default initialization arguments.
            Default: None.

    Returns:
        runner: The constructed runner.
    """
    if cfg.type == "runner_pooling":
        runner = Runner(cfg=cfg,
                        infer_dataset=infer_dataset,
                        infer_game=infer_game,
                        infer_features=infer_features,
                        infer_video=infer_video)
    elif cfg.type == "runner_CALF":
        runner = Runner(cfg=cfg,
                        infer_dataset=infer_dataset_CALF,
                        infer_game=None,
                        infer_features=None,
                        infer_video=None)
    elif cfg.type == "runner_JSON":
        runner = Runner(cfg=cfg,
                        infer_dataset=infer_dataset_JSON,
                        infer_game=infer_game,
                        infer_features=infer_features,
                        infer_video=infer_video) 
    else:
        runner = None
    return runner


class Runner():
    def __init__(self, cfg, 
                infer_dataset, 
                infer_game,
                infer_features,
                infer_video):
        self.infer_dataset = infer_dataset
        self.infer_game = infer_game
        self.infer_features = infer_features
        self.infer_video = infer_video