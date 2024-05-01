import os 
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import signal
import logging
from datetime import datetime
import time
from mmengine.config import Config, DictAction
import numpy as np
import torch
from oslactionspotting.apis.inference import build_inferer
from oslactionspotting.core.utils.dali import get_repartition_gpu
from oslactionspotting.core.utils.default_args import get_default_args_dataset
from oslactionspotting.core.utils.eval import search_best_epoch
from oslactionspotting.core.utils.io import check_config
from oslactionspotting.datasets.builder import build_dataloader, build_dataset
from oslactionspotting.models.builder import build_model


def parse_args():

    parser = ArgumentParser(description='context aware loss function', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")
    parser.add_argument("--checkpoint", type=str, help="path of model checkpoint")

    parser.add_argument("--overwrite", action="store_true", help="whether to overwrite the results")
    parser.add_argument("--confidence_threshold", type=float, default=0.5, help="confidence threshold for results")

    parser.add_argument("--cfg-options", nargs="+", action=DictAction, help="override settings")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
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
    # log_path = os.path.join(cfg.work_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S.log'))
    logging.basicConfig(
        level=numeric_level,
        format=
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[
            # logging.FileHandler(log_path),
            logging.StreamHandler()
        ])

    #Check configs files
    logging.info('Checking configs files')
    check_config(cfg)

    def whether_infer_split(cfg):
        if cfg.type == "SoccerNetGames" or cfg.type == "SoccerNetClipsTestingCALF" :
            if cfg.split == None :
                return False
            else : return True
        elif cfg.type == "FeatureVideosfromJSON" or cfg.type == "FeatureVideosChunksfromJson":
            if cfg.path.endswith('.json'):
                return True
            else : return False
        elif cfg.type == "VideoGameWithOpencvVideo" or cfg.type == 'VideoGameWithDaliVideo':
            if cfg.path.endswith('.json'):
                return True
            else : return False
        else :
            raise ValueError
        
    cfg.infer_split = whether_infer_split(cfg.dataset.test)
    dali = getattr(cfg, 'dali', False)
    if dali : cfg.repartitions = get_repartition_gpu()
    
    logging.info(cfg)
    
    # Start Timing
    start=time.time()
    logging.info('Starting main function')
    
    model = None
    # Ensure weights are not None
    if cfg.model.load_weights is None:
        if cfg.runner.type == "runner_e2e":
            best_epoch = search_best_epoch(cfg.work_dir)
            cfg.model.load_weights = os.path.join(cfg.work_dir, 'checkpoint_{:03d}.pt'.format(best_epoch))
        else:
            cfg.model.load_weights = os.path.join(cfg.work_dir, "model.pth.tar")
    
    # Build Model
    model = build_model(
        cfg, 
        verbose = False if cfg.runner.type == "runner_e2e" else True, 
        default_args={"classes":cfg.classes} if cfg.runner.type == "runner_e2e" else None)
    

    dataset_infer = build_dataset(cfg.dataset.test,cfg.training.GPU, get_default_args_dataset('test', cfg, cfg.runner.type == "runner_e2e", dali))

    logging.info('Build inferer')

    inferer = build_inferer(cfg,model)

    logging.info('Start inference')

    results = inferer.infer(dataset_infer)
    
    # # print results only if not done on full split
    # if cfg.runner.type == 'runner_e2e':
    #     print(f"Found {len(results[0]['events'])} actions!")
    # else:
    #     print(f"Found {len(results['predictions'])} actions!")

    return 


if __name__ == '__main__':
    main()