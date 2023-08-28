
import __future__

import numpy as np
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from .head import build_head
from .backbone import build_backbone
from .neck import build_neck


class LearnablePoolingModel(nn.Module):
    def __init__(self, weights=None, input_size=512, 
    num_classes=17, vocab_size=64, 
    window_size=15, framerate=2, 
    backbone="PreExtracted", 
    neck="NetVLAD++", 
    head="LinearLayer", 
    post_proc="NMS"):
        """
        INPUT: a Tensor of shape (batch_size,window_size,feature_size)
        OUTPUTS: a Tensor of shape (batch_size,num_classes+1)
        """

        super(LearnablePoolingModel, self).__init__()

        self.cfg_backbone = backbone
        self.cfg_neck = neck
        self.cfg_head = head
        self.cfg_post_proc = post_proc

        # self.backbone = backbone.type
        # self.neck = neck.type
        # self.head = head.type
        # self.post_proc = post_proc.type

        self.window_size_frame=window_size * framerate
        self.pooling_layer_input_dimension = input_size
        self.num_classes = num_classes
        self.framerate = framerate
        self.vlad_k = vocab_size
        
        # create backbone
        self.backbone = build_backbone(self.cfg_backbone)

        self.pooling_layer_input_dimension = 512

        # create neck
        # self.neck = build_neck(self.cfg_neck)
        
        if self.cfg_neck.type == "MaxPool":
            self.pooling_layer = nn.MaxPool1d(self.window_size_frame, stride=1)
            self.pooling_layer_output_dimension = self.pooling_layer_input_dimension
        
        if self.cfg_neck.type == "MaxPool++":
            self.pooling_layer_before = nn.MaxPool1d(int(self.window_size_frame/2), stride=1)
            self.pooling_layer_after = nn.MaxPool1d(int(self.window_size_frame/2), stride=1)
            self.pooling_layer_output_dimension = 2*self.pooling_layer_input_dimension
            

        if self.cfg_neck.type == "AvgPool":
            self.pooling_layer = nn.AvgPool1d(self.window_size_frame, stride=1)
            self.pooling_layer_output_dimension = self.pooling_layer_input_dimension

        if self.cfg_neck.type == "AvgPool++":
            self.pooling_layer_before = nn.AvgPool1d(int(self.window_size_frame/2), stride=1)
            self.pooling_layer_after = nn.AvgPool1d(int(self.window_size_frame/2), stride=1)
            self.pooling_layer_output_dimension = 2*self.pooling_layer_input_dimension
            

        elif self.cfg_neck.type == "NetVLAD":
            self.pooling_layer = NetVLAD_pool(cluster_size=self.vlad_k, feature_size=self.pooling_layer_input_dimension,
                                            add_batch_norm=True)
            self.pooling_layer_output_dimension = self.vlad_k*self.pooling_layer_input_dimension

        elif self.cfg_neck.type == "NetVLAD++":
            self.pooling_layer_before = NetVLAD_pool(cluster_size=int(self.vlad_k/2), feature_size=self.pooling_layer_input_dimension,
                                            add_batch_norm=True)
            self.pooling_layer_after = NetVLAD_pool(cluster_size=int(self.vlad_k/2), feature_size=self.pooling_layer_input_dimension,
                                            add_batch_norm=True)
            self.pooling_layer_output_dimension = self.vlad_k*self.pooling_layer_input_dimension



        elif self.cfg_neck.type == "NetRVLAD":
            self.pooling_layer = NetRVLAD_pool(cluster_size=self.vlad_k, feature_size=self.pooling_layer_input_dimension,
                                            add_batch_norm=True)
            self.pooling_layer_output_dimension = self.vlad_k*self.pooling_layer_input_dimension

        elif self.cfg_neck.type == "NetRVLAD++":
            self.pooling_layer_before = NetRVLAD_pool(cluster_size=int(self.vlad_k/2), feature_size=self.pooling_layer_input_dimension,
                                            add_batch_norm=True)
            self.pooling_layer_after = NetRVLAD_pool(cluster_size=int(self.vlad_k/2), feature_size=self.pooling_layer_input_dimension,
                                            add_batch_norm=True)
            self.pooling_layer_output_dimension = self.vlad_k*self.pooling_layer_input_dimension


        # Build Head
        self.head = build_head(self.cfg_head)
        

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

        inputs = self.backbone(inputs)

        # Temporal pooling operation
        if self.cfg_neck.type == "MaxPool" or self.cfg_neck.type == "AvgPool":
            inputs_pooled = self.pooling_layer(inputs.permute((0, 2, 1))).squeeze(-1)

        elif self.cfg_neck.type == "MaxPool++" or self.cfg_neck.type == "AvgPool++":
            nb_frames_50 = int(inputs.shape[1]/2)    
            input_before = inputs[:, :nb_frames_50, :]        
            input_after = inputs[:, nb_frames_50:, :]  
            inputs_before_pooled = self.pooling_layer_before(input_before.permute((0, 2, 1))).squeeze(-1)
            inputs_after_pooled = self.pooling_layer_after(input_after.permute((0, 2, 1))).squeeze(-1)
            inputs_pooled = torch.cat((inputs_before_pooled, inputs_after_pooled), dim=1)


        elif self.cfg_neck.type == "NetVLAD" or self.cfg_neck.type == "NetRVLAD":
            inputs_pooled = self.pooling_layer(inputs)

        elif self.cfg_neck.type == "NetVLAD++" or self.cfg_neck.type == "NetRVLAD++":
            nb_frames_50 = int(inputs.shape[1]/2)
            inputs_before_pooled = self.pooling_layer_before(inputs[:, :nb_frames_50, :])
            inputs_after_pooled = self.pooling_layer_after(inputs[:, nb_frames_50:, :])
            inputs_pooled = torch.cat((inputs_before_pooled, inputs_after_pooled), dim=1)


        # Extra FC layer with dropout and sigmoid activation
        output = self.head(inputs_pooled)

        return output

    def post_proc(self):

        return



import torch
import torch.nn as nn
import torch.nn.functional as F
# from sklearn.neighbors import NearestNeighbors
import numpy as np



from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch as th
import math


class NetVLAD_pool(nn.Module):
    def __init__(self, cluster_size, feature_size, add_batch_norm=True):
        super(NetVLAD_pool, self).__init__()
        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.clusters = nn.Parameter((1/math.sqrt(feature_size))
                *th.randn(feature_size, cluster_size))
        self.clusters2 = nn.Parameter((1/math.sqrt(feature_size))
                *th.randn(1, feature_size, cluster_size))

        self.add_batch_norm = add_batch_norm
        self.out_dim = cluster_size*feature_size

    def forward(self,x):
        # x [BS, T, D]
        max_sample = x.size()[1]

        # LOUPE
        if self.add_batch_norm: # normalization along feature dimension
            x = F.normalize(x, p=2, dim=2)

        x = x.reshape(-1,self.feature_size)
        assignment = th.matmul(x,self.clusters) 

        assignment = F.softmax(assignment,dim=1)
        assignment = assignment.view(-1, max_sample, self.cluster_size)

        a_sum = th.sum(assignment,-2,keepdim=True)
        a = a_sum*self.clusters2

        assignment = assignment.transpose(1,2)

        x = x.view(-1, max_sample, self.feature_size)
        vlad = th.matmul(assignment, x)
        vlad = vlad.transpose(1,2)
        vlad = vlad - a

        # L2 intra norm
        vlad = F.normalize(vlad)
        
        # flattening + L2 norm
        vlad = vlad.reshape(-1, self.cluster_size*self.feature_size)
        vlad = F.normalize(vlad)

        return vlad


class NetRVLAD_pool(nn.Module):
    def __init__(self, cluster_size, feature_size, add_batch_norm=True):
        super(NetRVLAD_pool, self).__init__()
        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.clusters = nn.Parameter((1/math.sqrt(feature_size))
                *th.randn(feature_size, cluster_size))
        # self.clusters2 = nn.Parameter((1/math.sqrt(feature_size))
        #         *th.randn(1, feature_size, cluster_size))
        # self.clusters = nn.Parameter(torch.rand(1,feature_size, cluster_size))
        # self.clusters2 = nn.Parameter(torch.rand(1,feature_size, cluster_size))

        self.add_batch_norm = add_batch_norm
        # self.batch_norm = nn.BatchNorm1d(cluster_size)
        self.out_dim = cluster_size*feature_size
        #  (+ 128 params?)
    def forward(self,x):
        max_sample = x.size()[1]

        # LOUPE
        if self.add_batch_norm: # normalization along feature dimension
            x = F.normalize(x, p=2, dim=2)

        x = x.reshape(-1,self.feature_size)
        assignment = th.matmul(x,self.clusters)

        assignment = F.softmax(assignment,dim=1)
        assignment = assignment.view(-1, max_sample, self.cluster_size)

        # a_sum = th.sum(assignment,-2,keepdim=True)
        # a = a_sum*self.clusters2

        assignment = assignment.transpose(1,2)

        x = x.view(-1, max_sample, self.feature_size)
        rvlad = th.matmul(assignment, x)
        rvlad = rvlad.transpose(-1,1)

        # vlad = vlad.transpose(1,2)
        # vlad = vlad - a

        # L2 intra norm
        rvlad = F.normalize(rvlad)
        
        # flattening + L2 norm
        rvlad = rvlad.reshape(-1, self.cluster_size*self.feature_size)
        rvlad = F.normalize(rvlad)

        return rvlad


if __name__ == "__main__":
    vlad = NetVLAD_pool(cluster_size=64, feature_size=512)

    feat_in = torch.rand((3,120,512))
    print("in", feat_in.shape)
    feat_out = vlad(feat_in)
    print("out", feat_out.shape)
    print(512*64)



    BS =256
    T = 15
    framerate= 2
    D = 512
    pool = "NetRVLAD++"
    model = NetVLAD(pool=pool, input_size=D, framerate=framerate, window_size=T)
    print(model)
    inp = torch.rand([BS,T*framerate,D])
    print(inp.shape)
    output = NetVLAD(inp)
    print(output.shape)