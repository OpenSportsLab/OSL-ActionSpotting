import pytorch_lightning as pl

import time
from SoccerNet.Evaluation.utils import AverageMeter

from oslspotting.core.loss import build_criterion
from oslspotting.core.scheduler import build_scheduler
from oslspotting.core.optimizer import build_optimizer

class LiteBaseModel(pl.LightningModule):
    def __init__(self,cfg_train):

        super().__init__()

        if cfg_train:
            self.criterion = build_criterion(cfg_train.criterion)
            
            self.cfg_train = cfg_train

            self.best_loss = 9e99
    
    def forward(self, inputs):
        return self.model(inputs)
    
    def on_train_epoch_start(self):
        self.batch_time,self.data_time,self.losses,self.end = self.pre_loop()
        
    def on_validation_epoch_start(self):
        self.batch_time,self.data_time,self.losses,self.end = self.pre_loop()

    def on_train_epoch_end(self):
        print('')
        self.losses_avg = self.losses.avg
    
    def on_fit_end(self) -> None:
        return self.best_state

    def configure_optimizers(self):
        self.optimizer = build_optimizer(self.parameters(), self.cfg_train.optimizer)
        self.scheduler = build_scheduler(self.optimizer, self.cfg_train.scheduler)
        return self.optimizer
    
    def pre_loop(self):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()
        return batch_time,data_time,losses,end