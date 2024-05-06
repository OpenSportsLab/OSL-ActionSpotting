
import __future__
from typing import Any

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from oslactionspotting.models.litebase import LiteBaseModel

import os

from oslactionspotting.datasets.utils import timestamps2long, batch2long
from SoccerNet.Downloader import getListGames

from oslactionspotting.models.utils import NMS, create_folders, get_json_data, predictions2json, predictions2json_runnerjson, zipResults

from .heads import build_head
from .backbones import build_backbone
from .necks import build_neck

class ContextAwareModel(nn.Module):
    def __init__(self, weights=None, 
                 backbone="PreExtracted", 
                neck="NetVLAD++", 
                head="LinearLayer", 
                post_proc="NMS"):
    # input_size=512, num_classes=3, 
    # chunk_size=240, dim_capsule=16,
    # receptive_field=80, num_detections=5, 
    # framerate=2):
        """
        INPUT: a Tensor of the form (batch_size,1,chunk_size,input_size)
        OUTPUTS:    1. The segmentation of the form (batch_size,chunk_size,num_classes)
                    2. The action spotting of the form (batch_size,num_detections,2+num_classes)
        """

        super(ContextAwareModel, self).__init__()


        # self.input_size = input_size
        # self.num_classes = num_classes
        # self.dim_capsule = dim_capsule
        # self.receptive_field = receptive_field
        # self.num_detections = num_detections
        # self.chunk_size = chunk_size
        # self.framerate = framerate

        # self.pyramid_size_1 = int(np.ceil(receptive_field/7))
        # self.pyramid_size_2 = int(np.ceil(receptive_field/3))
        # self.pyramid_size_3 = int(np.ceil(receptive_field/2))
        # self.pyramid_size_4 = int(np.ceil(receptive_field))

        # # Base Convolutional Layers
        # self.conv_1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(1,input_size))
        # self.conv_2 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1,1))

        # # Temporal Pyramidal Module
        # self.pad_p_1 = nn.ZeroPad2d((0,0,(self.pyramid_size_1-1)//2, self.pyramid_size_1-1-(self.pyramid_size_1-1)//2))
        # self.pad_p_2 = nn.ZeroPad2d((0,0,(self.pyramid_size_2-1)//2, self.pyramid_size_2-1-(self.pyramid_size_2-1)//2))
        # self.pad_p_3 = nn.ZeroPad2d((0,0,(self.pyramid_size_3-1)//2, self.pyramid_size_3-1-(self.pyramid_size_3-1)//2))
        # self.pad_p_4 = nn.ZeroPad2d((0,0,(self.pyramid_size_4-1)//2, self.pyramid_size_4-1-(self.pyramid_size_4-1)//2))
        # self.conv_p_1 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=(self.pyramid_size_1,1))
        # self.conv_p_2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(self.pyramid_size_2,1))
        # self.conv_p_3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(self.pyramid_size_3,1))
        # self.conv_p_4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(self.pyramid_size_4,1))

        # # -------------------
        # # Segmentation module
        # # -------------------

        # self.kernel_seg_size = 3
        # self.pad_seg = nn.ZeroPad2d((0,0,(self.kernel_seg_size-1)//2, self.kernel_seg_size-1-(self.kernel_seg_size-1)//2))
        # self.conv_seg = nn.Conv2d(in_channels=152, out_channels=dim_capsule*num_classes, kernel_size=(self.kernel_seg_size,1))
        # self.batch_seg = nn.BatchNorm2d(num_features=self.chunk_size, momentum=0.01,eps=0.001) 


        # # -------------------
        # # detection module
        # # -------------------       
        # self.max_pool_spot = nn.MaxPool2d(kernel_size=(3,1),stride=(2,1))
        # self.kernel_spot_size = 3
        # self.pad_spot_1 = nn.ZeroPad2d((0,0,(self.kernel_spot_size-1)//2, self.kernel_spot_size-1-(self.kernel_spot_size-1)//2))
        # self.conv_spot_1 = nn.Conv2d(in_channels=num_classes*(dim_capsule+1), out_channels=32, kernel_size=(self.kernel_spot_size,1))
        # self.max_pool_spot_1 = nn.MaxPool2d(kernel_size=(3,1),stride=(2,1))
        # self.pad_spot_2 = nn.ZeroPad2d((0,0,(self.kernel_spot_size-1)//2, self.kernel_spot_size-1-(self.kernel_spot_size-1)//2))
        # self.conv_spot_2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(self.kernel_spot_size,1))
        # self.max_pool_spot_2 = nn.MaxPool2d(kernel_size=(3,1),stride=(2,1))

        # # Confidence branch
        # self.conv_conf = nn.Conv2d(in_channels=16*(chunk_size//8-1), out_channels=self.num_detections*2, kernel_size=(1,1))

        # # Class branch
        # self.conv_class = nn.Conv2d(in_channels=16*(chunk_size//8-1), out_channels=self.num_detections*self.num_classes, kernel_size=(1,1))
        # self.softmax = nn.Softmax(dim=-1)

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

        # -----------------------------------
        # Feature input (chunks of the video)
        # -----------------------------------
        # input_shape: (batch,channel,frames,dim_features)
        #print("Input size: ", inputs.size())

        features = self.backbone(inputs)
        conv_seg, output_segmentation = self.neck(features)
        output_spotting = self.head(conv_seg, output_segmentation)
        return output_segmentation, output_spotting
        # -------------------------------------
        # Temporal Convolutional neural network
        # -------------------------------------


        # Base Convolutional Layers
        conv_1 = F.relu(self.conv_1(inputs))
        #print("Conv_1 size: ", conv_1.size())
        
        conv_2 = F.relu(self.conv_2(conv_1))
        #print("Conv_2 size: ", conv_2.size())


        # Temporal Pyramidal Module
        conv_p_1 = F.relu(self.conv_p_1(self.pad_p_1(conv_2)))
        #print("Conv_p_1 size: ", conv_p_1.size())
        conv_p_2 = F.relu(self.conv_p_2(self.pad_p_2(conv_2)))
        #print("Conv_p_2 size: ", conv_p_2.size())
        conv_p_3 = F.relu(self.conv_p_3(self.pad_p_3(conv_2)))
        #print("Conv_p_3 size: ", conv_p_3.size())
        conv_p_4 = F.relu(self.conv_p_4(self.pad_p_4(conv_2)))
        #print("Conv_p_4 size: ", conv_p_4.size())

        concatenation = torch.cat((conv_2,conv_p_1,conv_p_2,conv_p_3,conv_p_4),1)
        #print("Concatenation size: ", concatenation.size())


        # -------------------
        # Segmentation module
        # -------------------

        conv_seg = self.conv_seg(self.pad_seg(concatenation))
        #print("Conv_seg size: ", conv_seg.size())

        conv_seg_permuted = conv_seg.permute(0,2,3,1)
        #print("Conv_seg_permuted size: ", conv_seg_permuted.size())

        conv_seg_reshaped = conv_seg_permuted.view(conv_seg_permuted.size()[0],conv_seg_permuted.size()[1],self.dim_capsule,self.num_classes)
        #print("Conv_seg_reshaped size: ", conv_seg_reshaped.size())


        #conv_seg_reshaped_permuted = conv_seg_reshaped.permute(0,3,1,2)
        #print("Conv_seg_reshaped_permuted size: ", conv_seg_reshaped_permuted.size())

        conv_seg_norm = torch.sigmoid(self.batch_seg(conv_seg_reshaped))
        #print("Conv_seg_norm: ", conv_seg_norm.size())


        #conv_seg_norm_permuted = conv_seg_norm.permute(0,2,3,1)
        #print("Conv_seg_norm_permuted size: ", conv_seg_norm_permuted.size())

        output_segmentation = torch.sqrt(torch.sum(torch.square(conv_seg_norm-0.5), dim=2)*4/self.dim_capsule)
        #print("Output_segmentation size: ", output_segmentation.size())


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


        return output_segmentation, output_spotting

class LiteContextAwareModel(LiteBaseModel):
    def __init__(self, cfg=None, weights=None,
                 backbone="PreExtracted", 
                neck="CNN++", 
                head="SpottingCALF", 
                post_proc="NMS",
                runner="runner_CALF"):
                #  input_size=512, num_classes=3, 
                #  chunk_size=240, dim_capsule=16,
                #  receptive_field=80, num_detections=5, 
                #  framerate=2, runner="runner_CALF"):
        """
        INPUT: a Tensor of shape (batch_size,window_size,feature_size)
        OUTPUTS: a Tensor of shape (batch_size,num_classes+1)
        """
        super().__init__(cfg.training)

        # self.model=ContextAwareModel(weights,input_size,
        #                              num_classes,chunk_size,
        #                              dim_capsule,receptive_field,
        #                              num_detections,framerate)
        # check compatibility dims Backbone - Neck - Head
        assert(backbone.output_dim == neck.input_size)
        assert(neck.num_classes == head.num_classes)
        assert(neck.dim_capsule == head.dim_capsule)
        assert(neck.num_detections == head.num_detections)
        assert(neck.chunk_size == head.chunk_size)
        
        self.chunk_size = neck.chunk_size
        self.receptive_field = neck.receptive_field
        self.framerate = neck.framerate

        self.model=ContextAwareModel(weights,backbone,neck,head,post_proc)
        
        self.overwrite = True

        self.cfg = cfg

        self.runner = runner

        self.infer_split = getattr(cfg, 'infer_split', True)

    def process(self,labels,targets,feats):
        labels=labels.float()
        targets=targets.float()
        feats=feats.unsqueeze(1)
        return labels,targets,feats

    def _common_step(self, batch, batch_idx):
        feats,labels,targets=batch
        labels,targets,feats=self.process(labels,targets,feats)
        output_segmentation, output_spotting = self.forward(feats)
        return self.criterion([labels, targets], [output_segmentation, output_spotting]), feats.size(0)

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
    
    def on_predict_start(self):
        self.stop_predict = False

        if self.infer_split:
            self.output_folder, self.output_results, self.stop_predict = create_folders(self.cfg.dataset.test.results, self.cfg.work_dir, self.overwrite)
            if self.runner == "runner_JSON":
                self.target_dir = os.path.join(self.cfg.work_dir, self.output_folder)
            else:
                self.target_dir = self.output_results

        if not self.stop_predict:
            self.spotting_predictions = list()
            self.spotting_grountruth = list()
            self.spotting_grountruth_visibility = list()
            self.segmentation_predictions = list()
    
    def on_predict_end(self):
        if not self.stop_predict:
            # Transformation to numpy for evaluation
            targets_numpy = list()
            closests_numpy = list()
            detections_numpy = list()
            for target, detection in zip(self.spotting_grountruth_visibility,self.spotting_predictions):
                target_numpy = target.cpu().numpy()
                targets_numpy.append(target_numpy)
                detections_numpy.append(NMS(detection.numpy(), 20*self.framerate))
                closest_numpy = np.zeros(target_numpy.shape)-1
                #Get the closest action index
                for c in np.arange(target_numpy.shape[-1]):
                    indexes = np.where(target_numpy[:,c] != 0)[0].tolist()
                    if len(indexes) == 0 :
                        continue
                    indexes.insert(0,-indexes[0])
                    indexes.append(2*closest_numpy.shape[0])
                    for i in np.arange(len(indexes)-2)+1:
                        start = max(0,(indexes[i-1]+indexes[i])//2)
                        stop = min(closest_numpy.shape[0], (indexes[i]+indexes[i+1])//2)
                        closest_numpy[start:stop,c] = target_numpy[indexes[i],c]
                closests_numpy.append(closest_numpy)

            # Save the predictions to the json format
            # if save_predictions:
            if self.runner == "runner_CALF":
                list_game = self.trainer.predict_dataloaders.dataset.listGames
                for index in np.arange(len(list_game)):
                    json_data = get_json_data(list_game[index])
                    if self.infer_split :
                        os.makedirs(os.path.join(self.cfg.work_dir, self.output_folder, list_game[index]), exist_ok=True)
                        output_file = os.path.join(self.cfg.work_dir, self.output_folder, list_game[index], "results_spotting.json")
                    else : 
                        output_file = os.path.join(self.cfg.work_dir, f"{self.cfg.dataset.test.results}.json")
                    json_data = predictions2json(
                        detections_numpy[index*2], 
                        detections_numpy[(index*2)+1],
                        json_data,
                        output_file, 
                        self.framerate)
                    self.json_data = json_data
            elif self.runner == "runner_JSON":
                list_videos = self.trainer.predict_dataloaders.dataset.data_json["videos"]
                for index in np.arange(len(list_videos)):
                    video = list_videos[index]["path_features"]
                    
                    if self.infer_split :
                        video = os.path.splitext(video)[0]
                        os.makedirs(os.path.join(self.cfg.work_dir, self.output_folder, video), exist_ok=True)
                        output_file = os.path.join(self.cfg.work_dir, self.output_folder, video, "results_spotting.json")
                    else : 
                        output_file = os.path.join(self.cfg.work_dir, f"{self.cfg.dataset.test.results}.json")
                    
                    json_data = get_json_data(video)
                    json_data = predictions2json_runnerjson(
                        detections_numpy[index], 
                        json_data,
                        output_file, 
                        self.framerate, 
                        inverse_event_dictionary= self.trainer.predict_dataloaders.dataset.inverse_event_dictionary)
                    self.json_data = json_data
            if self.infer_split:
                zipResults(zip_path = self.output_results,
                       target_dir = os.path.join(self.cfg.work_dir, self.output_folder),
                       filename="results_spotting.json")
    
    def predict_step(self, batch):
        if not self.stop_predict:
            if self.runner == "runner_CALF":
                feat_half1, feat_half2, label_half1, label_half2 = batch

                label_half1 = label_half1.float().squeeze(0)
                label_half2 = label_half2.float().squeeze(0)

                feat_half1 = feat_half1.squeeze(0)
                feat_half2 = feat_half2.squeeze(0)

                feat_half1=feat_half1.unsqueeze(1)
                feat_half2=feat_half2.unsqueeze(1)

                # Compute the output
                output_segmentation_half_1, output_spotting_half_1 = self.forward(feat_half1)
                output_segmentation_half_2, output_spotting_half_2 = self.forward(feat_half2)

                timestamp_long_half_1 = timestamps2long(output_spotting_half_1.cpu().detach(), label_half1.size()[0], self.chunk_size, self.receptive_field)
                timestamp_long_half_2 = timestamps2long(output_spotting_half_2.cpu().detach(), label_half2.size()[0], self.chunk_size, self.receptive_field)
                segmentation_long_half_1 = batch2long(output_segmentation_half_1.cpu().detach(), label_half1.size()[0], self.chunk_size, self.receptive_field)
                segmentation_long_half_2 = batch2long(output_segmentation_half_2.cpu().detach(), label_half2.size()[0], self.chunk_size, self.receptive_field)

                self.spotting_grountruth.append(torch.abs(label_half1))
                self.spotting_grountruth.append(torch.abs(label_half2))
                self.spotting_grountruth_visibility.append(label_half1)
                self.spotting_grountruth_visibility.append(label_half2)
                self.spotting_predictions.append(timestamp_long_half_1)
                self.spotting_predictions.append(timestamp_long_half_2)
                self.segmentation_predictions.append(segmentation_long_half_1)
                self.segmentation_predictions.append(segmentation_long_half_2)
            elif self.runner == "runner_JSON":
                features, labels = batch

                labels = labels.float().squeeze(0)

                features = features.squeeze(0)

                features = features.unsqueeze(1)

                # Compute the output
                output_segmentation, output_spotting = self.forward(features)

                timestamp_long = timestamps2long(output_spotting.cpu().detach(), labels.size()[0], self.chunk_size, self.receptive_field)
                segmentation_long = batch2long(output_segmentation.cpu().detach(), labels.size()[0], self.chunk_size, self.receptive_field)

                self.spotting_grountruth.append(torch.abs(labels))
                self.spotting_grountruth_visibility.append(labels)
                self.spotting_predictions.append(timestamp_long)
                self.segmentation_predictions.append(segmentation_long)
