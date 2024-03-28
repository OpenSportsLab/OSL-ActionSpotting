import torch
from snspotting.models.e2espot import E2EModel
from .learnablepooling import LiteLearnablePoolingModel
from .contextaware import  LiteContextAwareModel

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
                  neck=cfg.model.neck, post_proc=cfg.model.post_proc, runner=cfg.runner.type)
    elif cfg.model.type == "ContextAware":
        model = LiteContextAwareModel(cfg = cfg, weights=cfg.model.load_weights,
                                input_size=cfg.model.input_size,
                                num_classes=cfg.model.num_classes,
                                chunk_size=cfg.model.chunk_size,
                                dim_capsule=cfg.model.dim_capsule,
                                receptive_field=cfg.model.receptive_field,
                                num_detections=cfg.model.num_detections,
                                framerate=cfg.model.framerate, runner = cfg.runner.type)
    elif cfg.model.type == "E2E":
        model = E2EModel(
            len(default_args["classes"]) + 1, cfg.model.feature_arch, cfg.model.temporal_arch,
            clip_len=cfg.dataset.clip_len, modality=cfg.dataset.modality,
            multi_gpu=cfg.model.multi_gpu)
        if cfg.model.load_weights != None:
            model.load(torch.load(cfg.model.load_weights))
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
