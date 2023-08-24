from .soccernet import SoccerNetClips, SoccerNetClipsTesting

import torch

def build_dataset(cfg, default_args=None):
    """Build a dataset from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        default_args (dict | None, optional): Default initialization arguments.
            Default: None.

    Returns:
        Dataset: The constructed dataset.
    """
    if cfg.type == "SoccerNetClips":
        dataset = SoccerNetClips(path=cfg.data_root, 
            features=cfg.features, split=cfg.split,
            version=2, framerate=2,
            window_size=cfg.window_size)
    elif cfg.type == "SoccerNetClipsTesting":
        dataset = SoccerNetClipsTesting(path=cfg.data_root, 
            features=cfg.features, split=cfg.split, 
            version=2, framerate=2,
            window_size=cfg.window_size)
    else:
        dataset=None
    return dataset


# def build_dataloader(dataset, batch_size, shuffle=True, max_num_worker=4, pin_memory=True):
def build_dataloader(dataset, cfg):
    """Build a dataloader from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        default_args (dict | None, optional): Default initialization arguments.
            Default: None.

    Returns:
        Dataloader: The constructed dataloader.
    """
    dataloader = torch.utils.data.DataLoader(dataset,
            batch_size=cfg.batch_size, shuffle=cfg.shuffle,
            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    return dataloader
