import os
import logging
from datetime import datetime
import signal
import time
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
from mmengine.config import Config, DictAction
from oslactionspotting.core.trainer import build_trainer
from oslactionspotting.core.utils.default_args import (
    get_default_args_dataset,
    get_default_args_model,
    get_default_args_train,
    get_default_args_trainer,
)

import random


from oslactionspotting.core.utils.io import check_config
from oslactionspotting.datasets.builder import build_dataloader, build_dataset
from oslactionspotting.models.builder import build_model


def parse_args():

    parser = ArgumentParser(
        description="Training script using config files.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")

    # not that important
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    # parser.add_argument("--id", type=int, default=0, help="repeat experiment id")
    parser.add_argument("--resume-from", type=str, default=None, help="resume from a checkpoint")
    # parser.add_argument("--ema", action="store_true", help="whether to use model EMA")
    # parser.add_argument("--wandb", action="store_true", help="whether to use wandb to log everything")
    # parser.add_argument("--not_eval", action="store_true", help="whether not to eval, only do inference")
    # parser.add_argument("--disable_deterministic", action="store_true", help="disable deterministic for faster speed")
    # parser.add_argument("--static_graph", action="store_true", help="set static_graph==True in DDP")
    parser.add_argument(
        "--cfg-options", nargs="+", action=DictAction, help="override settings"
    )

    # # parser.add_argument('--logging_dir',       required=False, type=str,   default="log", help='Where to log' )

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

    def set_seed(seed):
        random.seed(seed)  # Python random module
        np.random.seed(seed)  # NumPy
        torch.manual_seed(seed)  # PyTorch
        torch.cuda.manual_seed(seed)  # PyTorch CUDA
        torch.cuda.manual_seed_all(seed)  # Multi-GPU training

        # Ensures deterministic behavior
        torch.backends.cudnn.deterministic = True  
        torch.backends.cudnn.benchmark = False  

        # Ensures deterministic behavior for CUDA operations
        torch.use_deterministic_algorithms(True, warn_only=True)

    set_seed(args.seed)
    def signal_handler(signal, frame):
        print("\nScript aborted by user.")
        raise SystemExit

    # Set up the signal handler for KeyboardInterrupt because of pytorch lightning
    signal.signal(signal.SIGINT, signal_handler)

    # Create Work directory
    os.makedirs(cfg.work_dir, exist_ok=True)

    # Define logging
    numeric_level = getattr(logging, cfg.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % cfg.log_level)

    # Create logs folder
    os.makedirs(os.path.join(cfg.work_dir, "logs"), exist_ok=True)
    # Define logs folder
    log_path = os.path.join(
        cfg.work_dir, "logs", datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")
    )
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )

    # Check configs files
    logging.info("Checking configs files")
    check_config(cfg)

    # Dump configuration file
    cfg.dump(os.path.join(cfg.work_dir, "config.py"))
    logging.info(cfg)

    # Start Timing
    start = time.time()
    logging.info("Starting main function")

    # Build Model
    logging.info("Build Model")
    model = build_model(
        cfg,
        verbose=False if cfg.runner.type == "runner_e2e" else True,
        default_args=get_default_args_model(cfg),
    )
    # Build Datasets
    logging.info("Build Datasets")

    dataset_Train = build_dataset(
        cfg.dataset.train,
        cfg.training.GPU,
        get_default_args_dataset("train", cfg),
    )
    dataset_Valid = build_dataset(
        cfg.dataset.valid,
        cfg.training.GPU,
        get_default_args_dataset("valid", cfg),
    )

    # Build Dataloaders
    logging.info("Build Dataloaders")

    train_loader = build_dataloader(
        dataset_Train,
        cfg.dataset.train.dataloader,
        cfg.training.GPU,
        getattr(cfg, "dali", False),
    )
    valid_loader = build_dataloader(
        dataset_Valid,
        cfg.dataset.valid.dataloader,
        cfg.training.GPU,
        getattr(cfg, "dali", False),
    )

    # Build Trainer
    logging.info("Build Trainer")
    trainer = build_trainer(
        cfg.training,
        model,
        get_default_args_trainer(cfg, len(train_loader)),
        resume_from = args.resume_from
    )

    # Start training`
    logging.info("Start training")

    trainer.train(
        **get_default_args_train(
            model,
            train_loader,
            valid_loader,
            cfg.classes,
            cfg.training.type,
        )
    )

    logging.info(f"Total Execution Time is {time.time()-start} seconds")
    # return


if __name__ == "__main__":
    main()