
import __future__
from typing import Any

import numpy as np
import warnings
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from SoccerNet.Evaluation.utils import AverageMeter
import time

from .heads import build_head
from .backbones import build_backbone
from .necks import build_neck

from snspotting.core.optimizer import build_optimizer
from snspotting.core.scheduler import build_scheduler
from snspotting.core.loss import build_criterion

import logging

class LearnablePoolingModel(pl.LightningModule):
    def __init__(self, cfg_train=None, weights=None, 
                backbone="PreExtracted", 
                neck="NetVLAD++", 
                head="LinearLayer", 
                post_proc="NMS"):
        """
        INPUT: a Tensor of shape (batch_size,window_size,feature_size)
        OUTPUTS: a Tensor of shape (batch_size,num_classes+1)
        """
        super(LearnablePoolingModel, self).__init__()

        # check compatibility dims Backbone - Neck - Head
        assert(backbone.output_dim == neck.input_dim)
        assert(neck.output_dim == head.input_dim)

        # Build Backbone
        self.backbone = build_backbone(backbone)

        # Build Neck
        self.neck = build_neck(neck)
        
        # Build Head
        self.head = build_head(head)
        
        # load weight if needed
        self.load_weights(weights=weights)

        self.criterion = build_criterion(cfg_train.criterion)
        self.optimizer = build_optimizer(self.parameters(), cfg_train.optimizer)
        self.scheduler = build_scheduler(self.optimizer, cfg_train.scheduler)

        self.best_loss = 9e99

    def load_weights(self, weights=None):
        if(weights is not None):
            print("=> loading checkpoint '{}'".format(weights))
            checkpoint = torch.load(weights)
            self.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(weights, checkpoint['epoch']))

    def forward(self, inputs):
        # input_shape: (batch,frames,dim_features)
        features = self.backbone(inputs)
        feature_pooled = self.neck(features)
        output = self.head(feature_pooled)
        return output

    def post_proc(self):
        return

    def on_train_epoch_start(self):
        self.batch_time,self.data_time,self.losses,self.end = self.pre_loop(self.model,True)

    def training_step(self, batch, batch_idx):
        feats,labels=batch
        output = self.model(feats)
        loss = self.criterion(labels,output)
        self.log_dict({"loss":loss},on_step=True,on_epoch=True,prog_bar=True)
        self.losses.update(loss.item(), feats.size(0))
        return loss

    def on_train_epoch_end(self):
        print('')
        self.losses_avg = self.losses.avg

    def validation_step(self, batch, batch_idx):
        feats,labels=batch
        output = self.model(feats)
        val_loss = self.criterion(labels,output)
        self.log_dict({"val_loss":val_loss},on_step=False,on_epoch=True,prog_bar=True)
        self.losses.update(val_loss.item(), feats.size(0))
        return val_loss
    
    def on_fit_end(self) -> None:
        return self.best_state

    def on_validation_epoch_end(self):
        pass
    def configure_optimizers(self):
        return self.optimizer,self.scheduler
    
    def pre_loop(self):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()
        return batch_time,data_time,losses,end
    
from pytorch_lightning.callbacks.progress import TQDMProgressBar

class CustomProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, pl_module):
        # don't show the version number
        items = super().get_metrics(trainer,pl_module)
        items.pop("v_num", None)
        return items
    
class MyCallback(pl.Callback):
    def __init__(self):
        super().__init__()
    def on_validation_epoch_end(self, trainer, pl_module):
        loss_validation = pl_module.losses.avg
        state = {
                'epoch': trainer.current_epoch + 1,
                'state_dict': pl_module.model.state_dict(),
                'best_loss': pl_module.best_loss,
                'optimizer': pl_module.optimizer.state_dict(),
            }

        # remember best prec@1 and save checkpoint
        is_better = loss_validation < best_loss
        best_loss = min(loss_validation, best_loss)

        # Save the best model based on loss only if the evaluation frequency too long
        if is_better:
            pl_module.best_state = state
            # torch.save(state, best_model_path)

        # Reduce LR on Plateau after patience reached
        prevLR = self.optimizer.param_groups[0]['lr']
        self.scheduler.step(loss_validation)
        currLR = self.optimizer.param_groups[0]['lr']

        if (currLR is not prevLR and self.scheduler.num_bad_epochs == 0):
            logging.info("Plateau Reached!")
        if (prevLR < 2 * self.scheduler.eps and
            self.scheduler.num_bad_epochs >= self.scheduler.patience):
            logging.info("Plateau Reached and no more reduction -> Exiting Loop")
            self.should_stop=True