from .nll import NLLLoss


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
    # else:
    #     criterion = None
    return criterion