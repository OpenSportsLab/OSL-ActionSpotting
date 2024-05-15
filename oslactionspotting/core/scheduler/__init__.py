import torch
from torch.optim.lr_scheduler import (
    ChainedScheduler, LinearLR, CosineAnnealingLR)


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
    elif cfg.type == "ChainedSchedulerE2E":
        # Warmup schedule
        num_steps_per_epoch = default_args["len_train_loader"] // cfg.acc_grad_iter
        cosine_epochs = cfg.num_epochs - cfg.warm_up_epochs
        print('Using Linear Warmup ({}) + Cosine Annealing LR ({})'.format(
            cfg.warm_up_epochs, cosine_epochs))
        scheduler = ChainedScheduler([
            LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                     total_iters=cfg.warm_up_epochs * num_steps_per_epoch),
            CosineAnnealingLR(optimizer,
                              num_steps_per_epoch * cosine_epochs)])
    else:
        scheduler = None
    return scheduler
