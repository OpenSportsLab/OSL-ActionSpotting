import torch
import torch.nn as nn

from mmdet.registry import MODELS
from .utils import weighted_loss
####################################################################################################################################################

# Context-aware loss function

####################################################################################################################################################
@MODELS.register_module()
class ContextAwareLoss(torch.nn.Module):

    def __init__(self, K, hit_radius = 0.1, miss_radius = 0.9):

        super(ContextAwareLoss,self).__init__()

        self.K = K
        self.hit_radius = float(hit_radius)
        self.miss_radius = float(miss_radius)

    def forward(self, gt_label, pred_score):

        K = self.K
        hit_radius = self.hit_radius
        miss_radius = self.miss_radius

        zeros = torch.zeros(pred_score.size()).to(pred_score.device).type(torch.float)
        pred_score = 1.-pred_score
        
        case1 = self.DownStep(gt_label, K[0]) * torch.max(zeros, - torch.log(pred_score) + torch.log(zeros + miss_radius))
        case2 = self.Interval(gt_label, K[0], K[1]) * torch.max(zeros, - torch.log(pred_score + (1.-pred_score)*(self.PartialIdentity(gt_label,K[0],K[1])-K[0])/(K[1]-K[0])) + torch.log(zeros + miss_radius))
        case3 = self.Interval(gt_label, K[1], 0.) * zeros
        case4 = self.Interval(gt_label, 0., K[2]) * torch.max(zeros, - torch.log(1.-pred_score + pred_score*(self.PartialIdentity(gt_label,0.,K[2])-0.)/(K[2]-0.)) + torch.log(zeros + 1.-hit_radius))
        case5 = self.Interval(gt_label, K[2], K[3]) * torch.max(zeros, - torch.log(pred_score + (1.-pred_score)*(self.PartialIdentity(gt_label,K[2],K[3])-K[3])/(K[2]-K[3])) + torch.log(zeros + miss_radius))
        case6 = self.UpStep(gt_label, K[3]) * torch.max(zeros, - torch.log(pred_score) + torch.log(zeros + miss_radius))
        
        
        L = case1 + case2 + case3 + case4 + case5 + case6
        
        return torch.sum(L)

    def UpStep(self,x,a): #0 if x<a, 1 if x >= a

        return 1.-torch.max(0.*x,torch.sign(a-x))

    def DownStep(self,x,a): #1 if x < a, 0 if x >=a

        return torch.max(0.*x,torch.sign(a-x))

    def Interval(self,x,a,b): # 1 if a<= x < b, 0 otherwise
        
        return self.UpStep(x,a) * self.DownStep(x,b)

    def PartialIdentity(self,x,a,b):#a if x<a, x if a<= x <b, b if x >= b

        return torch.min(torch.max(x,0.*x+a),0.*x+b)