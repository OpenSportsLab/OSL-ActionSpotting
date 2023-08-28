def build_neck(cfg, default_args=None):
    """Build a neck from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        default_args (dict | None, optional): Default initialization arguments.
            Default: None.

    Returns:
        neck: The constructed neck.
    """
    if cfg.type == "NetVLAD":
        neck = None
    else:
        neck = None 

    return neck
