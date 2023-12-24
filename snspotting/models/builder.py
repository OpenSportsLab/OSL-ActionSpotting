from .learnablepooling import LearnablePoolingModel,LiteLearnablePoolingModel
from .contextaware import ContextAwareModel, LiteContextAwareModel

import logging

def build_model(cfg, verbose=True, default_args=None):
    """Build a model from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        default_args (dict | None, optional): Default initialization arguments.
            Default: None.

    Returns:
        Model: The constructed model.
    """
    if cfg.model.type == "LearnablePooling":
        model = LiteLearnablePoolingModel(cfg = cfg, weights=cfg.model.load_weights, 
                  backbone=cfg.model.backbone, head=cfg.model.head, 
                  neck=cfg.model.neck, post_proc=cfg.model.post_proc)
    elif cfg.model.type == "ContextAware":
        model = LiteContextAwareModel(cfg = cfg, weights=cfg.load_weights,
                                input_size=cfg.input_size,
                                num_classes=cfg.num_classes,
                                chunk_size=cfg.chunk_size,
                                dim_capsule=cfg.dim_capsule,
                                receptive_field=cfg.receptive_field,
                                num_detections=cfg.num_detections,
                                framerate=cfg.framerate)
    else:
        model = None 

    if verbose:    
        # Display info on model
        logging.info(model)
        total_params = sum(p.numel()
                        for p in model.parameters() if p.requires_grad)
        parameters_per_layer  = [p.numel() for p in model.parameters() if p.requires_grad]
        logging.info("Total number of parameters: " + str(total_params))

    return model
