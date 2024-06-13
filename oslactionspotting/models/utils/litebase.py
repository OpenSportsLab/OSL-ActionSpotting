import logging
import pytorch_lightning as pl

import time
from SoccerNet.Evaluation.utils import AverageMeter

from oslactionspotting.core.loss.builder import build_criterion
from oslactionspotting.core.scheduler.builder import build_scheduler
from oslactionspotting.core.optimizer.builder import build_optimizer


class LiteBaseModel(pl.LightningModule):
    """Parent class for lightning module.
    Args:
        cfg_train (dict): Dict of config.
    """

    def __init__(self, cfg_train):

        super().__init__()

        if cfg_train:
            logging.info('Build criterion')
            self.criterion = build_criterion(cfg_train.criterion)
            logging.info(self.criterion)
            self.cfg_train = cfg_train

            self.best_loss = 9e99

    def forward(self, inputs):
        return self.model(inputs)

    def on_train_epoch_start(self):
        self.batch_time, self.data_time, self.losses, self.end = self.pre_loop()

    def on_validation_epoch_start(self):
        self.batch_time, self.data_time, self.losses, self.end = self.pre_loop()

    def on_train_epoch_end(self):
        print("")
        self.losses_avg = self.losses.avg

    def on_fit_end(self) -> None:
        return self.best_state

    def configure_optimizers(self):
        logging.info('Build optimizer')
        self.optimizer = build_optimizer(self.parameters(), self.cfg_train.optimizer)
        logging.info(self.optimizer)
        logging.info('Build Scheduler')
        self.scheduler = build_scheduler(self.optimizer, self.cfg_train.scheduler)
        return self.optimizer

    def pre_loop(self):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()
        return batch_time, data_time, losses, end
