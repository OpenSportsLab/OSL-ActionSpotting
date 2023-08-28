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
    
    # Build Datasets    
    dataset_Train = build_dataset(cfg.dataset.train)
    dataset_Val = build_dataset(cfg.dataset.val)
    
    # Build Dataloaders
    train_loader = build_dataloader(dataset_Train, cfg.dataset.train.dataloader)
    val_loader = build_dataloader(dataset_Val, cfg.dataset.val.dataloader)

    # Build Trainer
    criterion = build_criterion(cfg.training.criterion)
    optimizer = build_optimizer(model.parameters(), cfg.training.optimizer)
    scheduler = build_scheduler(optimizer, cfg.training.scheduler)

    # Start training
    logging.info("start training")

    best_loss = 9e99

    # loop over epochs
    for epoch in range(cfg.training.max_epochs):
        best_model_path = os.path.join(cfg.work_dir, "model.pth.tar")

        # train for one epoch
        loss_training = train_one_epoch(train_loader, model, criterion,
                            optimizer, epoch + 1, backprop=True)

        # evaluate on validation set
        loss_validation = train_one_epoch(
            val_loader, model, criterion, optimizer, epoch + 1, backprop=False)

        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }

        # remember best prec@1 and save checkpoint
        is_better = loss_validation < best_loss
        best_loss = min(loss_validation, best_loss)

        # Save the best model based on loss only if the evaluation frequency too long
        if is_better:
            torch.save(state, best_model_path)

        # Test the model on the validation set
        if epoch % cfg.training.evaluation_frequency == 0 and epoch != 0:
            performance_validation = testClassication(
                val_loader,
                model,
                work_dir=cfg.work_dir)

            logging.info("Validation performance at epoch " +
                        str(epoch+1) + " -> " + str(performance_validation))

        # Reduce LR on Plateau after patience reached
        prevLR = optimizer.param_groups[0]['lr']
        scheduler.step(loss_validation)
        currLR = optimizer.param_groups[0]['lr']
        if (currLR is not prevLR and scheduler.num_bad_epochs == 0):
            logging.info("Plateau Reached!")

        if (prevLR < 2 * scheduler.eps and
                scheduler.num_bad_epochs >= scheduler.patience):
            logging.info(
                "Plateau Reached and no more reduction -> Exiting Loop")
            break

    # trainer(train_loader, val_loader, 
    #         model, optimizer, scheduler, criterion,
    #         model_name=config_filename,
    #         max_epochs=cfg.training.max_epochs, 
    #         evaluation_frequency=cfg.training.evaluation_frequency)

    return 


if __name__ == '__main__':
    main()