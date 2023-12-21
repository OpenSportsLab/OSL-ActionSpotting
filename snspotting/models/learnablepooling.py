
import __future__
import json
from typing import Any
import zipfile

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

import os
import logging

from SoccerNet.Evaluation.utils import AverageMeter, EVENT_DICTIONARY_V2, INVERSE_EVENT_DICTIONARY_V2
from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V1, INVERSE_EVENT_DICTIONARY_V1


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
    def __init__(self, cfg_train=None, cfg=None, weights=None, 
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

        self.confidence_threshold = 0.0

        self.overwrite = True

        self.cfg = cfg

        self.stop_predict = False

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

    def on_predict_start(self) -> None:
        # Create folder name and zip file name
        self.output_folder=f"results_spotting_{'_'.join(self.cfg.dataset.test.split)}"
        self.output_results=os.path.join(self.cfg.work_dir, f"{self.output_folder}.zip")

        # Prevent overwriting existing results
        if os.path.exists(self.output_results) and not self.overwrite:
            logging.warning("Results already exists in zip format. Use [overwrite=True] to overwrite the previous results.The inference will not run over the previous results.")
            self.stop_predict=True
            # return output_results

        if not self.stop_predict:
            self.spotting_predictions = list()

    def on_predict_end(self):
        if not self.stop_predict:
            zipResults(zip_path = self.output_results,
                target_dir = os.path.join(self.cfg.work_dir, self.output_folder),
                filename="results_spotting.json")
    
    def predict_step(self, batch, batch_idx):
        if not self.stop_predict:
            game_ID, feat_half1, feat_half2, label_half1, label_half2 = batch

            game_ID = game_ID[0]
            feat_half1 = feat_half1.squeeze(0)
            feat_half2 = feat_half2.squeeze(0)

            # Compute the output for batches of frames
            BS = 256
            timestamp_long_half_1 = self.timestamp_half(feat_half1,BS)
            timestamp_long_half_2 = self.timestamp_half(feat_half2,BS)
            
            timestamp_long_half_1 = timestamp_long_half_1[:, 1:]
            timestamp_long_half_2 = timestamp_long_half_2[:, 1:]

            self.spotting_predictions.append(timestamp_long_half_1)
            self.spotting_predictions.append(timestamp_long_half_2)

            framerate = self.trainer.predict_dataloaders.dataset.framerate
            get_spot = get_spot_from_NMS

            json_data = get_json_data(False,game_ID=game_ID)

            for half, timestamp in enumerate([timestamp_long_half_1, timestamp_long_half_2]):
                for l in range(self.trainer.predict_dataloaders.dataset.num_classes):
                    spots = get_spot(
                        timestamp[:, l], window=self.cfg.model.post_proc.NMS_window*self.cfg.model.backbone.framerate, thresh=self.cfg.model.post_proc.NMS_threshold)
                    for spot in spots:
                        # print("spot", int(spot[0]), spot[1], spot)
                        frame_index = int(spot[0])
                        confidence = spot[1]
                        if confidence < self.confidence_threshold:
                            continue
                        
                        json_data["predictions"].append(get_prediction_data(False,frame_index,framerate,half=half,version=self.trainer.predict_dataloaders.dataset.version,l=l,confidence=confidence))
            
                json_data["predictions"] = sorted(json_data["predictions"], key=lambda x: int(x['position']))
                json_data["predictions"] = sorted(json_data["predictions"], key=lambda x: int(x['half']))

            os.makedirs(os.path.join(self.cfg.work_dir, self.output_folder, game_ID), exist_ok=True)
            with open(os.path.join(self.cfg.work_dir, self.output_folder, game_ID, "results_spotting.json"), 'w') as output_file:
                json.dump(json_data, output_file, indent=4)

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
