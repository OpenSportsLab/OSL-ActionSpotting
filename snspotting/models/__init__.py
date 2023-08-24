from .netvlad import NetVLAD

def build_model(cfg, default_args=None):
    """Build a model from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        default_args (dict | None, optional): Default initialization arguments.
            Default: None.

    Returns:
        Model: The constructed model.
    """
    if cfg.type == "NetVLAD":
        model = NetVLAD(weights=cfg.load_weights, input_size=cfg.feature_dim,
                  num_classes=len(cfg.classes), window_size=cfg.window_size, 
                  vocab_size = cfg.vocab_size,
                  framerate=2, pool=cfg.neck)
    # else:
    #     model = None 
    return model
