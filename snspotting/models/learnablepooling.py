
import __future__

import numpy as np
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

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
