import os
import logging
from datetime import datetime
import time
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch

from mmengine.config import Config, DictAction


from snspotting.loss import build_criterion# , #NLLLoss
from snspotting.datasets import build_dataset, build_dataloader
from snspotting.models import build_model, NetVLAD
from snspotting.engine import build_optimizer, build_scheduler
from snspotting.engine import trainer, test, testSpotting


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

    # Define logging
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)

    # Define output folder
    os.makedirs(os.path.join("models", cfg.engine.model_name), exist_ok=True)
    log_path = os.path.join("models", cfg.engine.model_name,
                            datetime.now().strftime('%Y-%m-%d_%H-%M-%S.log'))
    logging.basicConfig(
        level=numeric_level,
        format=
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ])

    # define GPUs
    if cfg.engine.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.engine.GPU)


    start=time.time()
    logging.info('Starting main function')

    print(cfg)
    
    
    dataset_Train = build_dataset(cfg.dataset.train)
    dataset_Val = build_dataset(cfg.dataset.val)
    # dataset_Test = build_dataset(cfg.dataset.test)
    # dataset_Train = SoccerNetClips(path=cfg.data_root, features=args.features, split=args.split_train, version=args.version, framerate=args.framerate, window_size=args.window_size)
    # dataset_Valid = SoccerNetClips(path=cfg.data_root, features=args.features, split=args.split_valid, version=args.version, framerate=args.framerate, window_size=args.window_size)
    # dataset_Valid_metric = SoccerNetClips(path=cfg.data_root, features=args.features, split=args.split_valid, version=args.version, framerate=args.framerate, window_size=args.window_size)
    # dataset_Test  = SoccerNetClipsTesting(path=cfg.data_root, features=args.features, split=args.split_test, version=args.version, framerate=args.framerate, window_size=args.window_size)

    # create dataloader
    train_loader = build_dataloader(dataset_Train, cfg.dataset.train.dataloader)
    val_loader = build_dataloader(dataset_Val, cfg.dataset.val.dataloader)


    # if args.feature_dim is None:
    #     args.feature_dim = dataset_Test[0][1].shape[-1]
    #     print("feature_dim found:", args.feature_dim)
    

    # build model
    model = build_model(cfg.model).cuda()
    # model = NetVLAD(weights=cfg.model.load_weights, input_size=cfg.model.feature_dim,
    #               num_classes=dataset_Train.num_classes, window_size=cfg.model.window_size, 
    #               vocab_size = cfg.model.vocab_size,
    #               framerate=2, pool=cfg.model.neck).cuda()
    
    # Display info on model
    logging.info(model)
    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    parameters_per_layer  = [p.numel() for p in model.parameters() if p.requires_grad]
    logging.info("Total number of parameters: " + str(total_params))

    
    # if not args.test_only:
    #     train_loader = torch.utils.data.DataLoader(dataset_Train,
    #         batch_size=args.batch_size, shuffle=True,
    #         num_workers=args.max_num_worker, pin_memory=True)

    #     val_loader = torch.utils.data.DataLoader(dataset_Val,
    #         batch_size=args.batch_size, shuffle=False,
    #         num_workers=args.max_num_worker, pin_memory=True)

    #     # val_metric_loader = torch.utils.data.DataLoader(dataset_Valid_metric,
    #     #     batch_size=args.batch_size, shuffle=False,
    #     #     num_workers=args.max_num_worker, pin_memory=True)


    # training parameters
    # if not args.t0est_only:
    criterion = build_criterion(cfg.engine.criterion)
    optimizer = build_optimizer(model.parameters(), cfg.engine.optimizer)
    scheduler = build_scheduler(optimizer, cfg.engine.scheduler)

    

    # start training
    trainer(train_loader, val_loader, 
            model, optimizer, scheduler, criterion,
            model_name=cfg.engine.model_name,
            max_epochs=cfg.engine.max_epochs, 
            evaluation_frequency=cfg.engine.evaluation_frequency)

    # For the best model only
    checkpoint = torch.load(os.path.join("models", cfg.engine.model_name, "model.pth.tar"))
    model.load_state_dict(checkpoint['state_dict'])

    # test on multiple splits [test/challenge]
    for split in cfg.test.split:
        dataset_Test = build_dataset(cfg.dataset.test)
        test_loader = build_dataloader(cfg.dataset.test.dataloader)

        # dataset_Test  = SoccerNetClipsTesting(path=cfg.data_root, features=args.features, split=[split], version=args.version, framerate=args.framerate, window_size=args.window_size)

        # test_loader = torch.utils.data.DataLoader(dataset_Test,
        #     batch_size=1, shuffle=False,
        #     num_workers=1, pin_memory=True)

        results = testSpotting(test_loader, model=model, model_name=cfg.engine.model_name, NMS_window=args.NMS_window, NMS_threshold=args.NMS_threshold)
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

def parse_args():

    parser = ArgumentParser(description='context aware loss function', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")

    # not that important
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--id", type=int, default=0, help="repeat experiment id")
    parser.add_argument("--resume", type=str, default=None, help="resume from a checkpoint")
    parser.add_argument("--ema", action="store_true", help="whether to use model EMA")
    parser.add_argument("--wandb", action="store_true", help="whether to use wandb to log everything")
    parser.add_argument("--not_eval", action="store_true", help="whether not to eval, only do inference")
    parser.add_argument("--disable_deterministic", action="store_true", help="disable deterministic for faster speed")
    parser.add_argument("--static_graph", action="store_true", help="set static_graph==True in DDP")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, help="override settings")


    # # parser.add_argument('--SoccerNet_path',   required=False, type=str,   default="/path/to/SoccerNet/",     help='Path for SoccerNet' )
    # # parser.add_argument('--features',   required=False, type=str,   default="ResNET_TF2.npy",     help='Video features' )
    # parser.add_argument('--max_epochs',   required=False, type=int,   default=1000,     help='Maximum number of epochs' )
    # parser.add_argument('--load_weights',   required=False, type=str,   default=None,     help='weights to load' )
    # parser.add_argument('--model_name',   required=False, type=str,   default="NetVLAD++",     help='named of the model to save' )
    # parser.add_argument('--test_only',   required=False, action='store_true',  help='Perform testing only' )

    # parser.add_argument('--split_train', nargs='+', default=["train"], help='list of split for training')
    # parser.add_argument('--split_valid', nargs='+', default=["valid"], help='list of split for validation')
    # parser.add_argument('--split_test', nargs='+', default=["test", "challenge"], help='list of split for testing')

    # parser.add_argument('--version', required=False, type=int,   default=2,     help='Version of the dataset' )
    # parser.add_argument('--feature_dim', required=False, type=int,   default=None,     help='Number of input features' )
    # parser.add_argument('--evaluation_frequency', required=False, type=int,   default=10,     help='Number of chunks per epoch' )
    # parser.add_argument('--framerate', required=False, type=int,   default=2,     help='Framerate of the input features' )
    # parser.add_argument('--window_size', required=False, type=int,   default=15,     help='Size of the chunk (in seconds)' )
    # parser.add_argument('--pool',       required=False, type=str,   default="NetVLAD++", help='How to pool' )
    # parser.add_argument('--vocab_size',       required=False, type=int,   default=64, help='Size of the vocabulary for NetVLAD' )
    # parser.add_argument('--NMS_window',       required=False, type=int,   default=30, help='NMS window in second' )
    # parser.add_argument('--NMS_threshold',       required=False, type=float,   default=0.0, help='NMS threshold for positive results' )

    # parser.add_argument('--batch_size', required=False, type=int,   default=256,     help='Batch size' )
    # parser.add_argument('--LR',       required=False, type=float,   default=1e-03, help='Learning Rate' )
    # parser.add_argument('--LRe',       required=False, type=float,   default=1e-06, help='Learning Rate end' )
    # parser.add_argument('--patience', required=False, type=int,   default=10,     help='Patience before reducing LR (ReduceLROnPlateau)' )

    # parser.add_argument('--GPU',        required=False, type=int,   default=-1,     help='ID of the GPU to use' )
    # parser.add_argument('--max_num_worker',   required=False, type=int,   default=4, help='number of worker to load data')
    # # parser.add_argument('--seed',   required=False, type=int,   default=0, help='seed for reproducibility')

    # # parser.add_argument('--logging_dir',       required=False, type=str,   default="log", help='Where to log' )
    parser.add_argument('--loglevel',   required=False, type=str,   default='INFO', help='logging level')

    # read args
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()