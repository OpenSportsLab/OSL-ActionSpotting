
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


class MaxPool(torch.nn.Module):
    def __init__(self, nb_frames):
        super(MaxPool, self).__init__()
        self.pooling_layer = nn.MaxPool1d(nb_frames, stride=1)

    def forward(self, inputs):
        return self.pooling_layer(inputs.permute((0, 2, 1))).squeeze(-1)


class MaxPool_temporally_aware(torch.nn.Module):
    def __init__(self, nb_frames):
        super(MaxPool_temporally_aware, self).__init__()
        self.pooling_layer_before = nn.MaxPool1d(int(nb_frames/2), stride=1)
        self.pooling_layer_after = nn.MaxPool1d(int(nb_frames/2), stride=1)

    def forward(self, inputs):
        nb_frames_50 = int(inputs.shape[1]/2)    
        input_before = inputs[:, :nb_frames_50, :]        
        input_after = inputs[:, nb_frames_50:, :]  
        inputs_before_pooled = self.pooling_layer_before(input_before.permute((0, 2, 1))).squeeze(-1)
        inputs_after_pooled = self.pooling_layer_after(input_after.permute((0, 2, 1))).squeeze(-1)
        inputs_pooled = torch.cat((inputs_before_pooled, inputs_after_pooled), dim=1)
        return inputs_pooled


class AvgPool(torch.nn.Module):
    def __init__(self, nb_frames):
        super(AvgPool, self).__init__()
        self.pooling_layer = nn.AvgPool1d(nb_frames, stride=1)

    def forward(self, inputs):
        return self.pooling_layer(inputs.permute((0, 2, 1))).squeeze(-1)
        

class AvgPool_temporally_aware(torch.nn.Module):
    def __init__(self, nb_frames):
        super(AvgPool_temporally_aware, self).__init__()
        self.pooling_layer_before = nn.AvgPool1d(int(nb_frames/2), stride=1)
        self.pooling_layer_after = nn.AvgPool1d(int(nb_frames/2), stride=1)

    def forward(self, inputs):
        nb_frames_50 = int(inputs.shape[1]/2)    
        input_before = inputs[:, :nb_frames_50, :]        
        input_after = inputs[:, nb_frames_50:, :]  
        inputs_before_pooled = self.pooling_layer_before(input_before.permute((0, 2, 1))).squeeze(-1)
        inputs_after_pooled = self.pooling_layer_after(input_after.permute((0, 2, 1))).squeeze(-1)
        inputs_pooled = torch.cat((inputs_before_pooled, inputs_after_pooled), dim=1)
        return inputs_pooled


class NetRVLAD(torch.nn.Module):
    def __init__(self, vocab_size, input_dim):
        super(NetRVLAD, self).__init__()
        self.pooling_layer = NetRVLAD_core(cluster_size=vocab_size, 
                                        feature_size=input_dim,
                                        add_batch_norm=True)

    def forward(self, inputs):
        return self.pooling_layer(inputs)
        

class NetRVLAD_temporally_aware(torch.nn.Module):
    def __init__(self, vocab_size, input_dim):
        super(NetRVLAD_temporally_aware, self).__init__()
        self.pooling_layer_before = NetRVLAD_core(cluster_size=int(vocab_size/2), 
                                            feature_size=input_dim,
                                            add_batch_norm=True)
        self.pooling_layer_after = NetRVLAD_core(cluster_size=int(vocab_size/2), 
                                            feature_size=input_dim,
                                            add_batch_norm=True)
                                        
    def forward(self, inputs):
        nb_frames_50 = int(inputs.shape[1]/2)
        inputs_before_pooled = self.pooling_layer_before(inputs[:, :nb_frames_50, :])
        inputs_after_pooled = self.pooling_layer_after(inputs[:, nb_frames_50:, :])
        inputs_pooled = torch.cat((inputs_before_pooled, inputs_after_pooled), dim=1)
        return inputs_pooled


class NetVLAD(torch.nn.Module):
    def __init__(self, vocab_size, input_dim):
        super(NetVLAD, self).__init__()
        self.pooling_layer = NetVLAD_core(cluster_size=vocab_size, 
                                        feature_size=input_dim,
                                        add_batch_norm=True)

    def forward(self, inputs):
        return self.pooling_layer(inputs)
        

class NetVLAD_temporally_aware(torch.nn.Module):
    def __init__(self, vocab_size, input_dim):
        super(NetVLAD_temporally_aware, self).__init__()
        self.pooling_layer_before = NetVLAD_core(cluster_size=int(vocab_size/2), 
                                            feature_size=input_dim,
                                            add_batch_norm=True)
        self.pooling_layer_after = NetVLAD_core(cluster_size=int(vocab_size/2), 
                                            feature_size=input_dim,
                                            add_batch_norm=True)
                                        
    def forward(self, inputs):
        nb_frames_50 = int(inputs.shape[1]/2)
        inputs_before_pooled = self.pooling_layer_before(inputs[:, :nb_frames_50, :])
        inputs_after_pooled = self.pooling_layer_after(inputs[:, nb_frames_50:, :])
        inputs_pooled = torch.cat((inputs_before_pooled, inputs_after_pooled), dim=1)
        return inputs_pooled



class NetVLAD_core(nn.Module):
    def __init__(self, cluster_size, feature_size, add_batch_norm=True):
        super(NetVLAD_core, self).__init__()
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


class NetRVLAD_core(nn.Module):
    def __init__(self, cluster_size, feature_size, add_batch_norm=True):
        super(NetRVLAD_core, self).__init__()
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
