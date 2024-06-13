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
        optimizer = torch.optim.Adam(
            parameters,
            lr=cfg.lr,
            betas=cfg.betas,
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
            amsgrad=cfg.amsgrad,
        )
    if cfg.type == "AdamWithScaler":
        optimizer = (
            torch.optim.AdamW(parameters, lr=cfg.learning_rate),
            torch.cuda.amp.GradScaler(),
        )
    return optimizer
