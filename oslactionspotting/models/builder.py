import torch
from oslactionspotting.models.e2espot import E2EModel
from .learnablepooling import LiteLearnablePoolingModel
from .contextaware import LiteContextAwareModel
import logging


def build_model(cfg, verbose=True, default_args=None):
    """Build a model from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        verbose (bool): Whether to display infos of the model or not.
            Default: True.
        default_args (dict | None, optional): Default initialization arguments.
            Default: None.

    Returns:
        Model: The constructed model.
    """
    if cfg.model.type == "LearnablePooling":
        model = LiteLearnablePoolingModel(
            cfg=cfg,
            weights=cfg.model.load_weights,
            backbone=cfg.model.backbone,
            head=cfg.model.head,
            neck=cfg.model.neck,
            post_proc=cfg.model.post_proc,
            runner=cfg.runner.type,
        )
    elif cfg.model.type == "ContextAware":
        model = LiteContextAwareModel(
            cfg=cfg,
            weights=cfg.model.load_weights,
            backbone=cfg.model.backbone,
            head=cfg.model.head,
            neck=cfg.model.neck,
            runner=cfg.runner.type,
        )
    elif cfg.model.type == "E2E":
        model = E2EModel(
            cfg,
            len(default_args["classes"]) + 1,
            cfg.model.backbone,
            cfg.model.head,
            clip_len=cfg.dataset.clip_len,
            modality=cfg.dataset.modality,
            multi_gpu=cfg.model.multi_gpu,
        )
        
        # Load weights if specified
        if cfg.model.load_weights is not None:
            if verbose:
                logging.info(f"Loading weights from: {cfg.model.load_weights}")
            
            try:
                # Load checkpoint
                checkpoint = torch.load(cfg.model.load_weights)
                
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    if verbose:
                        logging.info("Found model_state_dict in checkpoint")
                else:
                    state_dict = checkpoint
                    if verbose:
                        logging.info("Using checkpoint as state dict directly")
                
                new_state_dict = {}
                for key in list(state_dict.keys()):
                    if key.startswith("_features"):
                        new_state_dict["backbone." + key] = state_dict[key]
                    elif key.startswith("_pred_fine"):
                        new_state_dict["head." + key] = state_dict[key]
                    else:
                        new_state_dict[key] = state_dict[key]
                
                # Load processed state dict
                model.load(new_state_dict)
                if verbose:
                    logging.info("Model weights loaded successfully")
                    
            except Exception as e:
                logging.error(f"Error loading weights: {str(e)}")
                raise
    else:
        model = None
        logging.warning(f"Unknown model type: {cfg.model.type}")

    if verbose and model is not None:
        # Display model info
        logging.info(model)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        parameters_per_layer = [
            p.numel() for p in model.parameters() if p.requires_grad
        ]
        logging.info(f"Total trainable parameters: {total_params:,}")
        logging.info(f"Number of parameter groups: {len(parameters_per_layer)}")

    return model