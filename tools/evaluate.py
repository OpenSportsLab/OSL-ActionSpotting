import os
import logging
from datetime import datetime
import signal
import time
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
import mmengine
from mmengine.config import Config, DictAction


from snspotting.core.utils.dali import get_repartition_gpu
from snspotting.core.utils.dataset import load_classes
from snspotting.core.utils.eval import search_best_epoch
from snspotting.core.utils.io import check_config
from snspotting.models import build_model

from snspotting.core import build_runner, build_evaluator
                
def parse_args():

    parser = ArgumentParser(description='context aware loss function', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")

    parser.add_argument("--eval_only", action="store_true", help="to only evaluate without infer")
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

    def signal_handler(signal, frame):
        print("\nScript aborted by user.")
        raise SystemExit

    # Set up the signal handler for KeyboardInterrupt
    signal.signal(signal.SIGINT, signal_handler)
    
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
    if 'GPU' in cfg.training.keys():
        if cfg.training.GPU >= 0:
            if not has_gpu:
                cfg.training.GPU = -1
        cfg_training_gpu = True
    else :
        cfg_training_gpu = None

    # if(cfg_training_gpu):
    #     logging.info('On GPU')
    # else:
    #     logging.info('On CPU')
    
    check_config(cfg)

    dali=False
    if 'dali' in cfg.keys():
        dali = True
        cfg.repartitions = get_repartition_gpu()

    # Display configuration file
    # cfg.dump(os.path.join(cfg.work_dir, 'config.py'))
    logging.info(cfg)

    # Start Timing
    start=time.time()
    logging.info('Starting main function')

    model = None
    if not args.eval_only:
        # Ensure weights are not None
        if cfg.model.load_weights is None:
            if cfg.runner.type == "runner_e2e":
                best_epoch = search_best_epoch(cfg.work_dir)
                logging.info(f"Best epoch : {str(best_epoch)}")
                cfg.model.load_weights = os.path.join(cfg.work_dir, 'checkpoint_{:03d}.pt'.format(best_epoch))
            else:
                cfg.model.load_weights = os.path.join(cfg.work_dir, "model.pth.tar")
    
        # Build Model
        model = build_model(cfg, 
                            verbose = False if cfg.runner.type == "runner_e2e" else True, 
                            default_args={"classes":cfg.classes} if cfg.runner.type == "runner_e2e" else None)
    
    # Build Evaluator
    logging.info('Build Evaluator')

    evaluator = build_evaluator(cfg=cfg, model=model)

    if not args.eval_only:
        #Start inference
        logging.info("Start inference")

        results = evaluator.infer(cfg.dataset.test)
    else:
        results = os.path.join(cfg.work_dir,cfg.dataset.test.results)
        if not os.path.exists(results):
            raise FileNotFoundError(f"The path '{results}' does not exist.")

    # Start evaluate`
    logging.info("Start evaluate")

    evaluator.evaluate(cfg.dataset.test, results)

    # evaluator.predict(model,)
    logging.info("Done evaluating")

    
    logging.info(f'Total Execution Time is {time.time()-start} seconds')

    return 


if __name__ == '__main__':
    main()