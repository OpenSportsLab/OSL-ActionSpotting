from .evaluate import Evaluator, evaluate_pred_JSON, evaluate_pred_SN, evaluate_pred_E2E


def build_evaluator(cfg, default_args=None):
    """Build a evaluator from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        default_args (dict | None, optional): Default initialization arguments.
            Default: None.

    Returns:
        evaluator: The constructed evaluator.
    """
    if cfg.runner.type == "runner_JSON":
        evaluator = Evaluator(cfg=cfg, evaluate_Spotting=evaluate_pred_JSON)
    elif cfg.runner.type == "runner_pooling":
        evaluator = Evaluator(cfg=cfg, evaluate_Spotting=evaluate_pred_SN)
    elif cfg.runner.type == "runner_CALF":
        evaluator = Evaluator(cfg=cfg, evaluate_Spotting=evaluate_pred_SN)
    elif cfg.runner.type == "runner_e2e":
        evaluator = Evaluator(cfg=cfg, evaluate_Spotting=evaluate_pred_E2E)

    return evaluator
