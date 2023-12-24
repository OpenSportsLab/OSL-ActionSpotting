
import __future__
import json
import time
from typing import Any
import zipfile

import numpy as np
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from snspotting.models.litebase import LiteBaseModel

import os
import logging

from SoccerNet.Evaluation.utils import AverageMeter, EVENT_DICTIONARY_V2, INVERSE_EVENT_DICTIONARY_V2
from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V1, INVERSE_EVENT_DICTIONARY_V1

from snspotting.datasets.soccernet import timestamps2long, batch2long
from SoccerNet.Downloader import getListGames

from snspotting.models.utils import create_folders


class ContextAwareModel(nn.Module):
    def __init__(self, weights=None, 
    input_size=512, num_classes=3, 
    chunk_size=240, dim_capsule=16,
    receptive_field=80, num_detections=5, 
    framerate=2):
        """
        INPUT: a Tensor of the form (batch_size,1,chunk_size,input_size)
        OUTPUTS:    1. The segmentation of the form (batch_size,chunk_size,num_classes)
                    2. The action spotting of the form (batch_size,num_detections,2+num_classes)
        """

        super(ContextAwareModel, self).__init__()


        self.input_size = input_size
        self.num_classes = num_classes
        self.dim_capsule = dim_capsule
        self.receptive_field = receptive_field
        self.num_detections = num_detections
        self.chunk_size = chunk_size
        self.framerate = framerate

        self.pyramid_size_1 = int(np.ceil(receptive_field/7))
        self.pyramid_size_2 = int(np.ceil(receptive_field/3))
        self.pyramid_size_3 = int(np.ceil(receptive_field/2))
        self.pyramid_size_4 = int(np.ceil(receptive_field))

        # Base Convolutional Layers
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(1,input_size))
        self.conv_2 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1,1))

        # Temporal Pyramidal Module
        self.pad_p_1 = nn.ZeroPad2d((0,0,(self.pyramid_size_1-1)//2, self.pyramid_size_1-1-(self.pyramid_size_1-1)//2))
        self.pad_p_2 = nn.ZeroPad2d((0,0,(self.pyramid_size_2-1)//2, self.pyramid_size_2-1-(self.pyramid_size_2-1)//2))
        self.pad_p_3 = nn.ZeroPad2d((0,0,(self.pyramid_size_3-1)//2, self.pyramid_size_3-1-(self.pyramid_size_3-1)//2))
        self.pad_p_4 = nn.ZeroPad2d((0,0,(self.pyramid_size_4-1)//2, self.pyramid_size_4-1-(self.pyramid_size_4-1)//2))
        self.conv_p_1 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=(self.pyramid_size_1,1))
        self.conv_p_2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(self.pyramid_size_2,1))
        self.conv_p_3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(self.pyramid_size_3,1))
        self.conv_p_4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(self.pyramid_size_4,1))

        # -------------------
        # Segmentation module
        # -------------------

        self.kernel_seg_size = 3
        self.pad_seg = nn.ZeroPad2d((0,0,(self.kernel_seg_size-1)//2, self.kernel_seg_size-1-(self.kernel_seg_size-1)//2))
        self.conv_seg = nn.Conv2d(in_channels=152, out_channels=dim_capsule*num_classes, kernel_size=(self.kernel_seg_size,1))
        self.batch_seg = nn.BatchNorm2d(num_features=self.chunk_size, momentum=0.01,eps=0.001) 


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
    def __init__(self, cfg_train=None, cfg=None, weights=None, 
                 input_size=512, num_classes=3, 
                 chunk_size=240, dim_capsule=16,
                 receptive_field=80, num_detections=5, 
                 framerate=2):
        """
        INPUT: a Tensor of shape (batch_size,window_size,feature_size)
        OUTPUTS: a Tensor of shape (batch_size,num_classes+1)
        """
        super().__init__(cfg_train)

        self.model=ContextAwareModel(weights,input_size,
                                     num_classes,chunk_size,
                                     dim_capsule,receptive_field,
                                     num_detections,framerate)
        
        self.overwrite = True

        self.cfg = cfg

        self.stop_predict = False
    
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
        self.output_folder, self.output_results, self.stop_predict = create_folders(self.cfg.dataset.test.split, self.cfg.work_dir, self.output_folder, self.overwrite)
        
        if not self.stop_predict:
            self.spotting_predictions = list()
            self.spotting_grountruth = list()
            self.spotting_grountruth_visibility = list()
            self.segmentation_predictions = list()
            self.chunk_size = self.model.chunk_size
            self.receptive_field = self.model.receptive_field
    
    def on_predict_end(self):
        if not self.stop_predict:
            # Transformation to numpy for evaluation
            targets_numpy = list()
            closests_numpy = list()
            detections_numpy = list()
            for target, detection in zip(self.spotting_grountruth_visibility,self.spotting_predictions):
                target_numpy = target.cpu().numpy()
                targets_numpy.append(target_numpy)
                detections_numpy.append(NMS(detection.numpy(), 20*self.model.framerate))
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
            list_game = getListGames(self.trainer.predict_dataloaders.dataset.split)
            for index in np.arange(len(list_game)):
                predictions2json(detections_numpy[index*2], detections_numpy[(index*2)+1],self.cfg.work_dir+"/"+self.output_folder+"/", list_game[index], self.model.framerate)
            zipResults(zip_path = self.output_results,
                       target_dir = os.path.join(self.cfg.work_dir, self.output_folder),
                       filename="results_spotting.json")
    
    def predict_step(self, batch):
        if not self.stop_predict:
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
    
    def timestamp_half(self,feat_half,BS):
        timestamp_long_half = []
        for b in range(int(np.ceil(len(feat_half)/BS))):
            start_frame = BS*b
            end_frame = BS*(b+1) if BS * \
                (b+1) < len(feat_half) else len(feat_half)
            feat = feat_half[start_frame:end_frame]
            output = self.model(feat).cpu().detach().numpy()
            timestamp_long_half.append(output)
        return np.concatenate(timestamp_long_half)

def get_spot_from_NMS(Input, window=60, thresh=0.0):
    detections_tmp = np.copy(Input)
    indexes = []
    MaxValues = []
    while(np.max(detections_tmp) >= thresh):

        # Get the max remaining index and value
        max_value = np.max(detections_tmp)
        max_index = np.argmax(detections_tmp)
        MaxValues.append(max_value)
        indexes.append(max_index)
        # detections_NMS[max_index,i] = max_value

        nms_from = int(np.maximum(-(window/2)+max_index,0))
        nms_to = int(np.minimum(max_index+int(window/2), len(detections_tmp)))
        detections_tmp[nms_from:nms_to] = -1
    
    return np.transpose([indexes, MaxValues])
    
        
def get_json_data(calf,game_info=None,game_ID=None):
    json_data = dict()
    json_data["UrlLocal"] = game_info if calf else game_ID
    json_data["predictions"] = list()
    return json_data

def get_prediction_data(calf,frame_index, framerate, class_index=None, confidence=None, half=None, l=None, version=None, half_1=None):
    seconds = int((frame_index//framerate)%60)
    minutes = int((frame_index//framerate)//60)

    prediction_data = dict()
    prediction_data["gameTime"] = (str(1 if half_1 else 2 ) + " - " + str(minutes) + ":" + str(seconds)) if calf else f"{half+1} - {minutes:02.0f}:{seconds:02.0f}"
    prediction_data["label"] = INVERSE_EVENT_DICTIONARY_V2[class_index if calf else l] if version == 2 else INVERSE_EVENT_DICTIONARY_V1[l]
    prediction_data["position"] = str(int((frame_index/framerate)*1000))
    prediction_data["half"] = str(1 if half_1 else 2) if calf else str(half+1)
    prediction_data["confidence"] = str(confidence)

    return prediction_data

def zipResults(zip_path, target_dir, filename="results_spotting.json"):            
    zipobj = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)
    rootlen = len(target_dir) + 1
    for base, dirs, files in os.walk(target_dir):
        for file in files:
            if file == filename:
                fn = os.path.join(base, file)
                zipobj.write(fn, fn[rootlen:])

def NMS(detections, delta):
    
    # Array to put the results of the NMS
    detections_tmp = np.copy(detections)
    detections_NMS = np.zeros(detections.shape)-1

    # Loop over all classes
    for i in np.arange(detections.shape[-1]):
        # Stopping condition
        while(np.max(detections_tmp[:,i]) >= 0):

            # Get the max remaining index and value
            max_value = np.max(detections_tmp[:,i])
            max_index = np.argmax(detections_tmp[:,i])

            detections_NMS[max_index,i] = max_value

            detections_tmp[int(np.maximum(-(delta/2)+max_index,0)): int(np.minimum(max_index+int(delta/2), detections.shape[0])) ,i] = -1

    return detections_NMS

def predictions2json(predictions_half_1, predictions_half_2, output_path, game_info, framerate=2):

    os.makedirs(output_path + game_info, exist_ok=True)
    output_file_path = output_path + game_info + "/results_spotting.json"

    frames_half_1, class_half_1 = np.where(predictions_half_1 >= 0)
    frames_half_2, class_half_2 = np.where(predictions_half_2 >= 0)
    
    json_data = get_json_data(True,game_info=game_info)
    
    for frame_index, class_index in zip(frames_half_1, class_half_1):

        confidence = predictions_half_1[frame_index, class_index]

        json_data["predictions"].append(get_prediction_data(True,frame_index,framerate,class_index=class_index,confidence=confidence,version=2,half_1=True))

    for frame_index, class_index in zip(frames_half_2, class_half_2):

        confidence = predictions_half_2[frame_index, class_index]

        json_data["predictions"].append(get_prediction_data(True,frame_index,framerate,class_index=class_index,confidence=confidence,version=2,half_1=False))
    
    with open(output_file_path, 'w') as output_file:
        json.dump(json_data, output_file, indent=4)