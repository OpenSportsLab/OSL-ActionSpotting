import os
import logging
from datetime import datetime
import signal
import time
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
from mmengine.config import Config, DictAction
from oslactionspotting.core.utils.dali import get_repartition_gpu
from oslactionspotting.core.utils.default_args import get_default_args_dataset, get_default_args_model, get_default_args_trainer


from oslactionspotting.core.utils.io import check_config
from oslactionspotting.datasets import build_dataset, build_dataloader
from oslactionspotting.models import build_model
from oslactionspotting.core import build_trainer 
        
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

    #Check configs files
    logging.info('Checking configs files')
    check_config(cfg)

    dali=False
    if 'dali' in cfg.keys() and cfg.dali == True:
        dali = True
        cfg.repartitions = get_repartition_gpu()
    
    # Dump configuration file
    cfg.dump(os.path.join(cfg.work_dir, 'config.py'))
    logging.info(cfg)

    # Start Timing
    start=time.time()
    logging.info('Starting main function')
    
    # Build Model
    logging.info('Build Model')
    model = build_model(cfg, verbose = False if cfg.runner.type == "runner_e2e" else True, default_args = get_default_args_model(cfg, cfg.runner.type == "runner_e2e"))
    
    # Build Datasets    
    logging.info('Build Datasets')

    dataset_Train = build_dataset(cfg.dataset.train,cfg.training.GPU, get_default_args_dataset('train', cfg, cfg.runner.type == "runner_e2e", dali))
    dataset_Val = build_dataset(cfg.dataset.val,cfg.training.GPU, get_default_args_dataset('val', cfg, cfg.runner.type == "runner_e2e", dali))
    dataset_Val_Frames = None
    if cfg.runner.type == "runner_e2e" and 'criterion_val' in cfg.training.keys() and cfg.training.criterion_val == 'map':
        dataset_Val_Frames = build_dataset(cfg.dataset.val_data_frames,None,get_default_args_dataset('val_data_frames', cfg, True, dali))
    
    # Build Dataloaders
    logging.info('Build Dataloaders')

    
    train_loader = build_dataloader(dataset_Train, cfg.dataset.train.dataloader,cfg.training.GPU, dali)
    val_loader = build_dataloader(dataset_Val, cfg.dataset.val.dataloader,cfg.training.GPU, dali)
    if dataset_Val_Frames is not None:
        val_frames_loader = build_dataloader( dataset_Val_Frames, cfg.dataset.val_data_frames.dataloader,cfg.training.GPU, dali)


    # Build Trainer
    logging.info('Build Trainer')
    trainer = build_trainer(cfg.training, model, get_default_args_trainer(cfg, cfg.runner.type == "runner_e2e", dali, len(train_loader)))

    # Start training`
    logging.info("Start training")

    if cfg.runner.type == "runner_e2e":
        trainer.train(train_loader,val_loader,dataset_Val_Frames,cfg.classes)
    else:
        trainer.fit(model,train_loader,val_loader)
    
    if cfg.runner.type != "runner_e2e":
        best_model = model.best_state

        logging.info("Done training")
        print(best_model.get("epoch"))
        torch.save(best_model, os.path.join(cfg.work_dir, "model.pth.tar"))

        logging.info('Model saved')
        logging.info(os.path.join(cfg.work_dir, "model.pth.tar"))

    logging.info(f'Total Execution Time is {time.time()-start} seconds')
    # return 


if __name__ == '__main__':
    main()