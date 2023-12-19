
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

class LearnablePoolingModel(nn.Module):
    def __init__(self, weights=None, 
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
        # self.load_weights(weights=weights)

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
    
class LiteLearnablePoolingModel(pl.LightningModule):
    def __init__(self, cfg_train=None, weights=None, 
                backbone="PreExtracted", 
                neck="NetVLAD++", 
                head="LinearLayer", 
                post_proc="NMS"):
        """
        INPUT: a Tensor of shape (batch_size,window_size,feature_size)
        OUTPUTS: a Tensor of shape (batch_size,num_classes+1)
        """
        super().__init__()

        self.model=LearnablePoolingModel(weights,backbone,neck,head,post_proc)

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