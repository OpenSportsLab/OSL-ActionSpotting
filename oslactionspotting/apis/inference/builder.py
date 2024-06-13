from .inference import Inferer, infer_E2E, infer_JSON, infer_SN


def build_inferer(cfg, model, default_args=None):
    """Build a inferer from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        model: The model that will be used to infer.
        default_args (dict | None, optional): Default initialization arguments.
            Default: None.

    Returns:
        inferer: The constructed inferer.
    """

    if cfg.runner.type == "runner_JSON":
        inferer = Inferer(cfg=cfg, model=model, infer_Spotting=infer_JSON)
    elif cfg.runner.type == "runner_pooling":
        inferer = Inferer(cfg=cfg, model=model, infer_Spotting=infer_SN)
    elif cfg.runner.type == "runner_CALF":
        inferer = Inferer(cfg=cfg, model=model, infer_Spotting=infer_SN)
    elif cfg.runner.type == "runner_e2e":
        inferer = Inferer(cfg=cfg, model=model, infer_Spotting=infer_E2E)

    return inferer
