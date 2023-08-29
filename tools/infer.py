from snspotting.apis import init_spotter, inference_spotter







# import logging
# from datetime import datetime
# import time
# import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# import torch
# import mmengine
from mmengine.config import Config, DictAction


# import json


# from snspotting.datasets import build_dataset, build_dataloader
# from snspotting.models import build_model
# from snspotting.loss import build_criterion
# from snspotting.core import build_optimizer, build_scheduler

# from snspotting.core.inference import infer_features, infer_game, infer_video
# from snspotting.core.evaluation import evaluate_Spotting #testClassication, testSpotting


def parse_args():

    parser = ArgumentParser(description='context aware loss function', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")
    parser.add_argument("--input", type=str, help="path of game/features/video")
    parser.add_argument("--checkpoint", type=str, help="path of model checkpoint")
    # parser.add_argument("--video", type=str, help="path of video")
    # parser.add_argument("--features", type=str, help="path of features")

    parser.add_argument("--overwrite", action="store_true", help="whether to overwrite the results")
    parser.add_argument("--confidence_threshold", type=float, default=0.5, help="confidence threshold for results")

    # not that important
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, help="override settings")

    # read args
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    # Read Config
    cfg = Config.fromfile(args.config)
    
    # overwrite cfg from args
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    # define GPUs
    if cfg.training.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.training.GPU)

    # build the model from a config file and a checkpoint file
    model = init_spotter(cfg, args.checkpoint, device='cuda:0')

    # test a single image and show the results
    results = inference_spotter(cfg, model, input=args.input)

    # print results
    print(f"Found {len(results['predictions'])} actions!")

    return 


if __name__ == '__main__':
    main()