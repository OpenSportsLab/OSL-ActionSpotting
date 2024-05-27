from .visualizer import Visualizer
def build_visualizer(cfg, Default_args = None):
    """Build a visualizer from config dict.

    Args:
        cfg (dict): Config dict.
        default_args (dict | None, optional): Default initialization arguments.
            Default: None.

    Returns:
        inferer: The constructed inferer.
    """
    visualizer = Visualizer(cfg)
    return visualizer