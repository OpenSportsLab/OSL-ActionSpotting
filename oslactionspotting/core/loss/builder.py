from oslactionspotting.core.loss.ce import CELoss
from .nll import NLLLoss
from .calf import ContextAwareLoss, SpottingLoss
from .combine import Combined2x


def build_criterion(cfg, default_args=None):
    """Build a criterion from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        default_args (dict | None, optional): Default initialization arguments.
            Default: None.

    Returns:
        criterion: The constructed criterion.
    """
    if cfg.type == "NLLLoss":
        criterion = NLLLoss()
    elif cfg.type == "ContextAwareLoss":
        criterion = ContextAwareLoss(
            K=cfg.K,
            framerate=cfg.framerate,
            hit_radius=cfg.hit_radius,
            miss_radius=cfg.miss_radius,
        )
    elif cfg.type == "SpottingLoss":
        criterion = SpottingLoss(
            lambda_coord=cfg.lambda_coord, lambda_noobj=cfg.lambda_noobj
        )
    elif cfg.type == "Combined2x":
        c_1 = build_criterion(cfg.loss_1)
        c_2 = build_criterion(cfg.loss_2)
        criterion = Combined2x(c_1, c_2, cfg.w_1, cfg.w_2)

    elif cfg.type == "CrossEntropyLoss":
        criterion = CELoss()
    # else:
    #     criterion = None
    return criterion
