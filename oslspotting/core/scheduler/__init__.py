import torch

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