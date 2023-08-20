import torch
import torch.nn as nn

from mmdet.registry import MODELS
from .utils import weighted_loss

@MODELS.register_module()
class SpottingLoss(torch.nn.Module):

    def __init__(self, lambda_coord, lambda_noobj):
        super(SpottingLoss,self).__init__()

        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj


    def forward(self,y_true, y_pred):
        y_pred = self.permute_ypred_for_matching(y_true,y_pred)
        loss = torch.sum(y_true[:,:,0]*self.lambda_coord*torch.square(y_true[:,:,1]-y_pred[:,:,1])  +  y_true[:,:,0]*torch.square(y_true[:,:,0]-y_pred[:,:,0]) +  (1-y_true[:,:,0])*self.lambda_noobj*torch.square(y_true[:,:,0]-y_pred[:,:,0]) +  y_true[:,:,0]*torch.sum(torch.square(y_true[:,:,2:]-y_pred[:,:,2:]),axis=-1)) #-y_true[:,:,0]*torch.sum(y_true[:,:,2:]*torch.log(y_pred[:,:,2:]),axis=-1)
        return loss


    def permute_ypred_for_matching(self, y_true, y_pred):
        
        alpha = y_true[:,:,0]
        x = y_true[:,:,1]
        p = y_pred[:,:,1]
        nb_pred = x.shape[-1]
        
        
        D = torch.abs(x.unsqueeze(-1).repeat(1,1,nb_pred) - p.unsqueeze(-2).repeat(1,nb_pred,1))
        D1 = 1-D
        Permut = 0*D
        
        alpha_filter = alpha.unsqueeze(-1).repeat(1,1,nb_pred)
        
        v_filter = alpha_filter
        h_filter = 0*v_filter + 1 
        D2 = v_filter * D1

        for i in range(nb_pred):
            D2 = v_filter * D2
            D2 = h_filter * D2
            A = torch.nn.functional.one_hot(torch.argmax(D2,axis=-1),nb_pred)
            B = v_filter * A * D2
            C = torch.nn.functional.one_hot(torch.argmax(B,axis=-2),nb_pred).permute(0, 2, 1)
            E = v_filter * A * C
            Permut = Permut + E
            v_filter = (1-torch.sum(Permut,axis=-1))*alpha
            v_filter = v_filter.unsqueeze(-1).repeat(1,1,nb_pred)
            h_filter = 1-torch.sum(Permut, axis=-2)
            h_filter = h_filter.unsqueeze(-2).repeat(1,nb_pred,1)
        
        v_filter = 1-alpha_filter
        D2 = v_filter * D1
        D2 = h_filter * D2
        
        for i in range(nb_pred):
            D2 = v_filter * D2
            D2 = h_filter * D2
            A = torch.nn.functional.one_hot(torch.argmax(D2,axis=-1),nb_pred)
            B = v_filter * A * D2
            C = torch.nn.functional.one_hot(torch.argmax(B,axis=-2),nb_pred).permute(0, 2, 1)
            E = v_filter * A * C
            Permut = Permut + E
            v_filter = (1-torch.sum(Permut,axis=-1))*(1-alpha) #here comes the change
            v_filter = v_filter.unsqueeze(-1).repeat(1,1,nb_pred)
            h_filter = 1-torch.sum(Permut, axis=-2)
            h_filter = h_filter.unsqueeze(-2).repeat(1,nb_pred,1)
        
        permutation = torch.argmax(Permut,axis=-1)
        permuted = torch.gather(y_pred, 1, permutation.unsqueeze(-1).repeat(1,1,y_true.shape[-1]))
            
        return permuted