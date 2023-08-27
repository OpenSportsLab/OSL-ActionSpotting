from .learnablepooling import LearnablePoolingModel

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
    if cfg.type == "LearnablePooling":
        model = LearnablePoolingModel(weights=cfg.load_weights, input_size=cfg.backbone.feature_dim,
                  num_classes=len(cfg.classes), window_size=cfg.window_size, 
                  vocab_size=cfg.vocab_size, framerate=cfg.framerate, 
                  backbone=cfg.backbone.type, neck=cfg.neck, head=cfg.head)
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
