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
from snspotting.core import build_trainer 


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

    # check if cuda available
    has_gpu=torch.cuda.is_available()
    if cfg.training.GPU >= 0:
        if not has_gpu:
            cfg.training.GPU = -1

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
    logging.info('Build Model')

    if cfg.training.GPU >=0 :
        model = build_model(cfg.model).cuda()
    else:
        model = build_model(cfg.model)
    
    # Build Datasets    
    logging.info('Build Datasets')
    dataset_Train = build_dataset(cfg.dataset.train)
    dataset_Val = build_dataset(cfg.dataset.val)
    
    # Build Dataloaders
    logging.info('Build Dataloaders')
    train_loader = build_dataloader(dataset_Train, cfg.dataset.train.dataloader,cfg.training.GPU)
    val_loader = build_dataloader(dataset_Val, cfg.dataset.val.dataloader,cfg.training.GPU)

    # Build Trainer
    logging.info('Build Trainer')
    trainer = build_trainer(cfg.training, model)

    # Start training`
    logging.info("Start training")

    best_model = trainer.train(train_loader, val_loader)
    logging.info("Done training")

    torch.save(best_model, 
               os.path.join(cfg.work_dir, "model.pth.tar"))

    logging.info('Model saved')
    logging.info(os.path.join(cfg.work_dir, "model.pth.tar"))

    return 


if __name__ == '__main__':
    main()