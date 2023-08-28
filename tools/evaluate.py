import os
import logging
from datetime import datetime
import time
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
import mmengine
from mmengine.config import Config, DictAction


from snspotting.datasets import build_dataset, build_dataloader
from snspotting.models import build_model
from snspotting.loss import build_criterion
from snspotting.core import build_optimizer, build_scheduler

from snspotting.core.training import train_one_epoch
from snspotting.core.evaluation import testClassication, testSpotting


def parse_args():

    parser = ArgumentParser(description='context aware loss function', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")

    # not that important
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    # parser.add_argument("--id", type=int, default=0, help="repeat experiment id")
    # parser.add_argument("--resume", type=str, default=None, help="resume from a checkpoint")
    # parser.add_argument("--ema", action="store_true", help="whether to use model EMA")
    # parser.add_argument("--wandb", action="store_true", help="whether to use wandb to log everything")
    # parser.add_argument("--not_eval", action="store_true", help="whether not to eval, only do inference")
    # parser.add_argument("--disable_deterministic", action="store_true", help="disable deterministic for faster speed")
    # parser.add_argument("--static_graph", action="store_true", help="set static_graph==True in DDP")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, help="override settings")

    # # parser.add_argument('--logging_dir',       required=False, type=str,   default="log", help='Where to log' )
    parser.add_argument('--loglevel',   required=False, type=str,   default='INFO', help='logging level')

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
    
    # for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create Work directory
    os.makedirs(cfg.work_dir, exist_ok=True)

    # Define logging
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)

    # Define output folder
    log_path = os.path.join(cfg.work_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S.log'))
    logging.basicConfig(
        level=numeric_level,
        format=
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ])

    # define GPUs
    if cfg.training.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.training.GPU)

    # Dump configuration file
    cfg.dump(os.path.join(cfg.work_dir, 'config.py'))
    logging.info(cfg)

    # Start Timing
    start=time.time()
    logging.info('Starting main function')

    # Build Model
    model = build_model(cfg.model).cuda()

    # For the best model only
    checkpoint = torch.load(os.path.join(cfg.work_dir, "model.pth.tar"))
    model.load_state_dict(checkpoint['state_dict'])

    # test on multiple splits [test/challenge]
    for split in cfg.dataset.test.split:
        dataset_Test = build_dataset(cfg.dataset.test)
        test_loader = build_dataloader(dataset_Test, cfg.dataset.test.dataloader)

        results = testSpotting(test_loader, model=model, work_dir=cfg.work_dir, 
        NMS_window=cfg.model.NMS_window, NMS_threshold=cfg.model.NMS_threshold)
        if results is None:
            continue

        a_mAP = results["a_mAP"]
        a_mAP_per_class = results["a_mAP_per_class"]
        a_mAP_visible = results["a_mAP_visible"]
        a_mAP_per_class_visible = results["a_mAP_per_class_visible"]
        a_mAP_unshown = results["a_mAP_unshown"]
        a_mAP_per_class_unshown = results["a_mAP_per_class_unshown"]

        logging.info("Best Performance at end of training ")
        logging.info("a_mAP visibility all: " +  str(a_mAP))
        logging.info("a_mAP visibility all per class: " +  str( a_mAP_per_class))
        logging.info("a_mAP visibility visible: " +  str( a_mAP_visible))
        logging.info("a_mAP visibility visible per class: " +  str( a_mAP_per_class_visible))
        logging.info("a_mAP visibility unshown: " +  str( a_mAP_unshown))
        logging.info("a_mAP visibility unshown per class: " +  str( a_mAP_per_class_unshown))
    
    
    logging.info(f'Total Execution Time is {time.time()-start} seconds')

    return 


if __name__ == '__main__':
    main()