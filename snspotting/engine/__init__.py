from .trainer import trainer, test, testSpotting

import torch

def build_optimizer(parameters, cfg, default_args=None):
    """Build a optimizer from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        default_args (dict | None, optional): Default initialization arguments.
            Default: None.

    Returns:
        optimizer: The constructed optimizer.
    """
    if cfg.type == "Adam":
        optimizer = torch.optim.Adam(parameters, lr=cfg.lr, 
                                betas=cfg.betas, eps=cfg.eps, 
                                weight_decay=cfg.weight_decay, amsgrad=cfg.amsgrad)
    return optimizer

def build_scheduler(optimizer, cfg, default_args=None):
    """Build a scheduler from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        default_args (dict | None, optional): Default initialization arguments.
            Default: None.

    Returns:
        scheduler: The constructed scheduler.
    """
    if cfg.type == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
        mode=cfg.mode, verbose=cfg.verbose, patience=cfg.patience)
    else:
        scheduler = None
    return scheduler