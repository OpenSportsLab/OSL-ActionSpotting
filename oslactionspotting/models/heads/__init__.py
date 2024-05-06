import torch 
import torch.nn.functional as F
import torch.nn as nn

from oslactionspotting.models.modules import *

def build_head(cfg, default_args=None):
    """Build a head from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        default_args (dict | None, optional): Default initialization arguments.
            Default: None.

    Returns:
        head: The constructed head.
    """
    if cfg.type == "LinearLayer":
        head = LinearLayerHead(input_dim=cfg.input_dim, 
                            output_dim=cfg.num_classes+1)
    elif cfg.type == "SpottingCALF":
        head = SpottingCALFHead(num_classes = cfg.num_classes, 
                                dim_capsule = cfg.dim_capsule, 
                                num_detections = cfg.num_detections, 
                                chunk_size = cfg.chunk_size)
    elif cfg.type in ['', 'gru', 'deeper_gru', 'mstcn', 'asformer']:
        head = TemporalE2EHead(cfg.type, cfg.feat_dim, cfg.num_classes)
    else:
        head = None 

    return head


class TemporalE2EHead(nn.Module):
    def __init__(self, temporal_arch, feat_dim, num_classes):
        super().__init__()
        # Prevent the GRU params from going too big (cap it at a RegNet-Y 800MF)
        MAX_GRU_HIDDEN_DIM = 768
        if 'gru' in temporal_arch:
            hidden_dim = feat_dim
            if hidden_dim > MAX_GRU_HIDDEN_DIM:
                hidden_dim = MAX_GRU_HIDDEN_DIM
                print('Clamped GRU hidden dim: {} -> {}'.format(
                    feat_dim, hidden_dim))
            if temporal_arch in ('gru', 'deeper_gru'):
                self._pred_fine = GRUPrediction(
                    feat_dim, num_classes, hidden_dim,
                    num_layers=3 if temporal_arch[0] == 'd' else 1)
            else:
                raise NotImplementedError(temporal_arch)
        elif temporal_arch == 'mstcn':
            self._pred_fine = TCNPrediction(feat_dim, num_classes, 3)
        elif temporal_arch == 'asformer':
            self._pred_fine = ASFormerPrediction(feat_dim, num_classes, 3)
        elif temporal_arch == '':
            self._pred_fine = FCPrediction(feat_dim, num_classes)
        
    def forward(self, inputs):
        return self._pred_fine(inputs)

class LinearLayerHead(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearLayerHead, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.drop = torch.nn.Dropout(p=0.4)
        self.head = torch.nn.Linear(input_dim, output_dim)
        self.sigm = torch.nn.Sigmoid()
        

    def forward(self, inputs):
        return self.sigm(self.head(self.drop(inputs)))

class SpottingCALFHead(torch.nn.Module):
    def __init__(self, num_classes, dim_capsule, num_detections, chunk_size):
        super(SpottingCALFHead, self).__init__()

        self.num_classes = num_classes
        self.dim_capsule = dim_capsule
        self.num_detections = num_detections
        self.chunk_size = chunk_size

        # -------------------
        # detection module
        # -------------------       
        self.max_pool_spot = nn.MaxPool2d(kernel_size=(3,1),stride=(2,1))
        self.kernel_spot_size = 3
        self.pad_spot_1 = nn.ZeroPad2d((0,0,(self.kernel_spot_size-1)//2, self.kernel_spot_size-1-(self.kernel_spot_size-1)//2))
        self.conv_spot_1 = nn.Conv2d(in_channels=num_classes*(dim_capsule+1), out_channels=32, kernel_size=(self.kernel_spot_size,1))
        self.max_pool_spot_1 = nn.MaxPool2d(kernel_size=(3,1),stride=(2,1))
        self.pad_spot_2 = nn.ZeroPad2d((0,0,(self.kernel_spot_size-1)//2, self.kernel_spot_size-1-(self.kernel_spot_size-1)//2))
        self.conv_spot_2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(self.kernel_spot_size,1))
        self.max_pool_spot_2 = nn.MaxPool2d(kernel_size=(3,1),stride=(2,1))

        # Confidence branch
        self.conv_conf = nn.Conv2d(in_channels=16*(chunk_size//8-1), out_channels=self.num_detections*2, kernel_size=(1,1))

        # Class branch
        self.conv_class = nn.Conv2d(in_channels=16*(chunk_size//8-1), out_channels=self.num_detections*self.num_classes, kernel_size=(1,1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, conv_seg, output_segmentation):
        # ---------------
        # Spotting module
        # ---------------

        # Concatenation of the segmentation score to the capsules
        output_segmentation_reverse = 1-output_segmentation
        #print("Output_segmentation_reverse size: ", output_segmentation_reverse.size())

        output_segmentation_reverse_reshaped = output_segmentation_reverse.unsqueeze(2)
        #print("Output_segmentation_reverse_reshaped size: ", output_segmentation_reverse_reshaped.size())


        output_segmentation_reverse_reshaped_permutted = output_segmentation_reverse_reshaped.permute(0,3,1,2)
        #print("Output_segmentation_reverse_reshaped_permutted size: ", output_segmentation_reverse_reshaped_permutted.size())

        concatenation_2 = torch.cat((conv_seg, output_segmentation_reverse_reshaped_permutted), dim=1)
        #print("Concatenation_2 size: ", concatenation_2.size())

        conv_spot = self.max_pool_spot(F.relu(concatenation_2))
        #print("Conv_spot size: ", conv_spot.size())

        conv_spot_1 = F.relu(self.conv_spot_1(self.pad_spot_1(conv_spot)))
        #print("Conv_spot_1 size: ", conv_spot_1.size())

        conv_spot_1_pooled = self.max_pool_spot_1(conv_spot_1)
        #print("Conv_spot_1_pooled size: ", conv_spot_1_pooled.size())

        conv_spot_2 = F.relu(self.conv_spot_2(self.pad_spot_2(conv_spot_1_pooled)))
        #print("Conv_spot_2 size: ", conv_spot_2.size())

        conv_spot_2_pooled = self.max_pool_spot_2(conv_spot_2)
        #print("Conv_spot_2_pooled size: ", conv_spot_2_pooled.size())

        spotting_reshaped = conv_spot_2_pooled.view(conv_spot_2_pooled.size()[0],-1,1,1)
        #print("Spotting_reshape size: ", spotting_reshaped.size())

        # Confindence branch
        conf_pred = torch.sigmoid(self.conv_conf(spotting_reshaped).view(spotting_reshaped.shape[0],self.num_detections,2))
        #print("Conf_pred size: ", conf_pred.size())

        # Class branch
        conf_class = self.softmax(self.conv_class(spotting_reshaped).view(spotting_reshaped.shape[0],self.num_detections,self.num_classes))
        #print("Conf_class size: ", conf_class.size())

        output_spotting = torch.cat((conf_pred,conf_class),dim=-1)
        #print("Output_spotting size: ", output_spotting.size())

        return output_spotting