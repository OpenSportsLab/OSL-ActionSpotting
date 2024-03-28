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


from snspotting.core.utils.dataset import load_classes
from snspotting.core.utils.io import load_json
from snspotting.models import build_model

from snspotting.core import build_runner, build_evaluator

def search_best_epoch(work_dir):
    loss = load_json(os.path.join(work_dir,'loss.json'))
    val_mAP = 0
    for epoch_loss in loss:
        if epoch_loss["val_mAP"] > val_mAP :
            val_mAP = epoch_loss["val_mAP"]
            epoch = epoch_loss["epoch"]
    return epoch
def check_config(cfg):
    if cfg.runner.type == "runner_e2e":
        assert cfg.dataset.modality in ['rgb']
        assert cfg.model.feature_arch in [
                # From torchvision
                'rn18',
                'rn18_tsm',
                'rn18_gsm',
                'rn50',
                'rn50_tsm',
                'rn50_gsm',

                # From timm (following its naming conventions)
                'rny002',
                'rny002_tsm',
                'rny002_gsm',
                'rny008',
                'rny008_tsm',
                'rny008_gsm',

                # From timm
                'convnextt',
                'convnextt_tsm',
                'convnextt_gsm'
            ]
        assert cfg.model.temporal_arch in ['', 'gru', 'deeper_gru', 'mstcn', 'asformer']
        assert cfg.dataset.batch_size % cfg.training.acc_grad_iter == 0
        assert cfg.training.criterion in ['map', 'loss']
        if cfg.training.start_val_epoch is None:
            cfg.training.start_val_epoch = cfg.training.num_epochs - cfg.training.base_num_val_epochs
        if cfg.dataset.crop_dim <= 0:
            cfg.dataset.crop_dim = None
        if os.path.isfile(cfg.classes):
            cfg.classes = load_classes(cfg.classes)
        for key,value in cfg.dataset.items():
            if key in ['train','val','val_data_frames','test','challenge']:
                pass
            else:
                cfg.dataset['train'][key] = value
                cfg.dataset['val'][key] = value
                cfg.dataset['val_data_frames'][key] = value
                cfg.dataset['test'][key] = value
                cfg.dataset['challenge'][key] = value
                
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
        
    def get_repartition_gpu():
        x = torch.cuda.device_count()
        print("Number of gpus:",x)
        if x==2: return [0,1],[0,1]
        elif x==3: return [0,1],[1,2]
        elif x>3: return [0,1,2,3],[0,2,1,3]
    
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

    # Ensure weights are not None
    if cfg.model.load_weights is None:
        if cfg.runner.type == "runner_e2e":
            cfg.model.load_weights = os.path.join(cfg.work_dir, 'checkpoint_{:03d}.pt'.format(search_best_epoch(cfg.work_dir)))
        else:
            cfg.model.load_weights = os.path.join(cfg.work_dir, "model.pth.tar")
    
    # Build Model
    model = build_model(cfg, 
                        verbose = False if cfg.runner.type == "runner_e2e" else True, 
                        default_args={"classes":cfg.classes} if cfg.runner.type == "runner_e2e" else None)
    # model = build_model(cfg)
    
    # Build Evaluator
    logging.info('Build Evaluator')

    evaluator = build_evaluator(cfg=cfg, model=model)


    # Start evaluate`
    logging.info("Start evaluate")

    evaluator.evaluate(cfg.dataset.test)

    # evaluator.predict(model,)
    logging.info("Done evaluating")

    
    logging.info(f'Total Execution Time is {time.time()-start} seconds')

    return 


if __name__ == '__main__':
    main()