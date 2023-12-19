
import __future__
from typing import Any

import numpy as np
import warnings
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler

import torch
import torch.nn as nn
import torch.nn.functional as F

from snspotting.models.litebase import LiteBaseModel

from .heads import build_head
from .backbones import build_backbone
from .necks import build_neck

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
        self.load_weights(weights=weights)

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


class LiteLearnablePoolingModel(LiteBaseModel):
    def __init__(self, cfg_train=None, weights=None, 
                backbone="PreExtracted", 
                neck="NetVLAD++", 
                head="LinearLayer", 
                post_proc="NMS"):
        """
        INPUT: a Tensor of shape (batch_size,window_size,feature_size)
        OUTPUTS: a Tensor of shape (batch_size,num_classes+1)
        """
        super().__init__(cfg_train)

        self.model=LearnablePoolingModel(weights,backbone,neck,head,post_proc)

    def _common_step(self, batch, batch_idx):
        feats,labels=batch
        output = self.model(feats)
        return self.criterion(labels,output),feats.size(0)

    def training_step(self, batch, batch_idx):
        loss, size = self._common_step(batch,batch_idx)
        self.log_dict({"loss":loss},on_step=True,on_epoch=True,prog_bar=True)
        self.losses.update(loss.item(), size)
        return loss

    def validation_step(self, batch, batch_idx):
        val_loss, size = self._common_step(batch,batch_idx)
        self.log_dict({"val_loss":val_loss},on_step=False,on_epoch=True,prog_bar=True)
        self.losses.update(val_loss.item(), size)
        return val_loss