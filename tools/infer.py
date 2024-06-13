from datetime import datetime
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import signal
import logging
import time
from mmengine.config import Config, DictAction
import numpy as np
import torch
from oslactionspotting.apis.inference.builder import build_inferer
from oslactionspotting.apis.inference.utils import search_best_epoch
from oslactionspotting.core.utils.default_args import (
    get_default_args_dataset,
    get_default_args_model,
)
from oslactionspotting.core.utils.io import check_config, whether_infer_split
from oslactionspotting.datasets.builder import build_dataset
from oslactionspotting.models.builder import build_model


def parse_args():

    parser = ArgumentParser(
        description="context aware loss function",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")

    parser.add_argument(
        "--cfg-options", nargs="+", action=DictAction, help="override settings"
    )

    parser.add_argument("--seed", type=int, default=42, help="random seed")

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

    cfg.infer_split = whether_infer_split(cfg.dataset.test)

    logging.info(cfg)

    # Start Timing
    start = time.time()
    logging.info("Starting main function")

    model = None
    # Ensure weights are not None
    if cfg.model.load_weights is None:
        if cfg.runner.type == "runner_e2e":
            best_epoch = search_best_epoch(cfg.work_dir)
            cfg.model.load_weights = os.path.join(
                cfg.work_dir, "checkpoint_{:03d}.pt".format(best_epoch)
            )
        else:
            cfg.model.load_weights = os.path.join(cfg.work_dir, "model.pth.tar")

    # Build Model
    model = build_model(
        cfg,
        verbose=False if cfg.runner.type == "runner_e2e" else True,
        default_args=get_default_args_model(cfg),
    )

    dataset_infer = build_dataset(
        cfg.dataset.test, cfg.training.GPU, get_default_args_dataset("test", cfg)
    )

    logging.info("Build inferer")

    inferer = build_inferer(cfg, model)

    logging.info("Start inference")

    results = inferer.infer(dataset_infer)

    # logging.info(f'Predictions saved to {cfg.dataset.test.results if cfg.}')
    # # print results only if not done on full split
    # if cfg.runner.type == 'runner_e2e':
    #     print(f"Found {len(results[0]['events'])} actions!")
    # else:
    #     print(f"Found {len(results['predictions'])} actions!")

    return


if __name__ == "__main__":
    main()
