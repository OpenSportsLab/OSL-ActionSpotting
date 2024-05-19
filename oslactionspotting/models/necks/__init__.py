from .pooling import *
from .calf import *


def build_neck(cfg, default_args=None):
    """Build a neck from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        default_args (dict | None, optional): Default initialization arguments.
            Default: None.

    Returns:
        neck: The constructed neck.
    """
    if cfg.type == "MaxPool":
        neck = MaxPool(nb_frames=cfg.nb_frames)
    elif cfg.type == "MaxPool++":
        neck = MaxPool_temporally_aware(nb_frames=cfg.nb_frames)
    elif cfg.type == "AvgPool":
        neck = AvgPool(nb_frames=cfg.nb_frames)
    elif cfg.type == "AvgPool++":
        neck = AvgPool_temporally_aware(nb_frames=cfg.nb_frames)
    elif cfg.type == "NetRVLAD":
        neck = NetRVLAD(
            vocab_size=cfg.vocab_size,
            input_dim=cfg.input_dim,
        )
    elif cfg.type == "NetRVLAD++":
        neck = NetRVLAD_temporally_aware(
            vocab_size=cfg.vocab_size,
            input_dim=cfg.input_dim,
        )
    elif cfg.type == "NetVLAD":
        neck = NetVLAD(
            vocab_size=cfg.vocab_size,
            input_dim=cfg.input_dim,
        )
    elif cfg.type == "NetVLAD++":
        neck = NetVLAD_temporally_aware(
            vocab_size=cfg.vocab_size,
            input_dim=cfg.input_dim,
        )
    elif cfg.type == "CNN++":
        neck = CNN_temporally_aware(
            input_size=cfg.input_size,
            num_classes=cfg.num_classes,
            chunk_size=cfg.chunk_size,
            dim_capsule=cfg.dim_capsule,
            receptive_field=cfg.receptive_field,
            num_detections=cfg.num_detections,
            framerate=cfg.framerate,
        )
    else:
        neck = None

    return neck
