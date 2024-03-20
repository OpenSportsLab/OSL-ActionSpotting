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


from snspotting.datasets import build_dataset, build_dataloader
from snspotting.models import build_model
from snspotting.core import build_trainer 


import json

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
            if key in ['train','val','val_data_frames']:
                pass
            else:
                cfg.dataset['train'][key] = value
                cfg.dataset['val'][key] = value
                cfg.dataset['val_data_frames'][key] = value
        
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

def store_config(file_path, args, num_epochs, classes):
    config = {
        'dali' : args.dali,
        'dataset': "soccernetv2",
        'num_classes': len(classes),
        'modality': args.modality,
        'feature_arch': args.model.feature_arch,
        'temporal_arch': args.model.temporal_arch,
        'clip_len': args.clip_len,
        'batch_size': args.dataset.batch_size,
        'crop_dim': args.dataset.crop_dim,
        'num_epochs': num_epochs,
        'warm_up_epochs': args.warm_up_epochs,
        'learning_rate': args.learning_rate,
        'start_val_epoch': args.start_val_epoch,
        'gpu_parallel': args.model.multi_gpu,
        'epoch_num_frames': args.dataset.epoch_num_frames,
        #   EPOCH_NUM_FRAMES,
        'dilate_len': args.dataset.dilate_len,
        'mixup': args.dataset.mixup,
    }
    store_json(file_path, config, pretty=True)
def store_json(fpath, obj, pretty=False):
    kwargs = {}
    if pretty:
        kwargs['indent'] = 2
        kwargs['sort_keys'] = True
    with open(fpath, 'w') as fp:
        json.dump(obj, fp, **kwargs)
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
    else :
        cfg_training_gpu = None

    # # define GPUs
    # if cfg.training.GPU >= 0:
    #     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #     os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.training.GPU)

    def get_repartition_gpu():
        x = torch.cuda.device_count()
        print("Number of gpus:",x)
        if x==2: return [0,1],[0,1]
        elif x==3: return [0,1],[1,2]
        elif x>3: return [0,1,2,3],[0,2,1,3]
    
    check_config(cfg)

    if 'dali' in cfg.keys():
        dali = True
        cfg.repartitions = get_repartition_gpu()

    # Dump configuration file
    cfg.dump(os.path.join(cfg.work_dir, 'config.py'))
    logging.info(cfg)

    # Start Timing
    start=time.time()
    logging.info('Starting main function')
    
    # Build Datasets    
    logging.info('Build Datasets')

    
    dataset_Train = build_dataset(cfg.dataset.train,cfg.training.GPU if cfg_training_gpu is not None else None,{"classes":cfg.classes,'train' : True, 'acc_grad_iter' : cfg.training.acc_grad_iter, 'num_epochs' : cfg.training.num_epochs, 'repartitions' : cfg.repartitions} if dali else {'train' : True})
    dataset_Val = build_dataset(cfg.dataset.val,cfg.training.GPU if cfg_training_gpu is not None else None,{"classes":cfg.classes,'train' : False, 'acc_grad_iter' : cfg.training.acc_grad_iter, 'num_epochs' : cfg.training.num_epochs, 'repartitions' : cfg.repartitions} if dali else {'train' : False})
    if cfg.runner.type == "runner_e2e" and 'criterion' in cfg.training.keys() and cfg.training.criterion == 'map':
        dataset_Val_Frames = build_dataset(cfg.dataset.val_data_frames,None,{"classes":cfg.classes, 'repartitions' : cfg.repartitions})
    
    # Build Dataloaders
    logging.info('Build Dataloaders')
    if dali:
        train_loader = dataset_Train
        val_loader = dataset_Val
    else:
        train_loader = build_dataloader(dataset_Train, cfg.dataset.train.dataloader,cfg.training.GPU)
        val_loader = build_dataloader(dataset_Val, cfg.dataset.val.dataloader,cfg.training.GPU)

    # Build Model
    logging.info('Build Model')
    model = build_model(cfg, verbose = False if cfg.runner.type == "runner_e2e" else True, default_args={"classes":cfg.classes} if cfg.runner.type == "runner_e2e" else None)

    # Build Trainer
    logging.info('Build Trainer')
    trainer = build_trainer(cfg.training, model if cfg.runner.type == "runner_e2e" else None, {"len_train_loader":len(train_loader), 'work_dir' : cfg.work_dir, 'dali' : cfg.dali, 'modality' : cfg.dataset.modality, 'clip_len' : cfg.dataset.clip_len , 'crop_dim' : cfg.dataset.crop_dim, 'labels_dir' : cfg.labels_dir} if cfg.runner.type == "runner_e2e" else None)

    # Start training`
    logging.info("Start training")

    if cfg.runner.type == "runner_e2e":
        trainer.train(train_loader,val_loader,dataset_Val_Frames,cfg.classes)
    else:
        trainer.fit(model,train_loader,val_loader)
    # best_model = model.best_state

    # logging.info("Done training")
    # print(best_model.get("epoch"))
    # torch.save(best_model, 
    #            os.path.join(cfg.work_dir, "model.pth.tar"))

    # logging.info('Model saved')
    # logging.info(os.path.join(cfg.work_dir, "model.pth.tar"))

    # return 


if __name__ == '__main__':
    main()