
import __future__
import json
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from snspotting.core.runner import timestamp_half

from snspotting.models.litebase import LiteBaseModel
from snspotting.models.utils import create_folders, get_json_data, get_prediction_data, get_spot_from_NMS, zipResults

from .heads import build_head
from .backbones import build_backbone
from .necks import build_neck

import os

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
    def __init__(self, cfg=None, weights=None, 
                backbone="PreExtracted", 
                neck="NetVLAD++", 
                head="LinearLayer", 
                post_proc="NMS",
                runner="runner_pooling"):
        """
        INPUT: a Tensor of shape (batch_size,window_size,feature_size)
        OUTPUTS: a Tensor of shape (batch_size,num_classes+1)
        """
        super().__init__(cfg.training)

        self.model=LearnablePoolingModel(weights,backbone,neck,head,post_proc)

        self.confidence_threshold = 0.0

        self.overwrite = True

        self.cfg = cfg

        self.runner = runner

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

    def on_predict_start(self):
        self.output_folder, self.output_results, self.stop_predict = create_folders(self.cfg.dataset.test.split, self.cfg.work_dir, self.overwrite)
        
        if self.runner == "runner_JSON":
            self.target_dir = os.path.join(self.cfg.work_dir, self.output_folder)
        else:
            self.target_dir = self.output_results
            
        print(self.output_folder,self.output_results,self.target_dir)
        if not self.stop_predict:
            self.spotting_predictions = list()

    def on_predict_end(self):
        if not self.stop_predict:
            zipResults(zip_path = self.output_results,
                target_dir = os.path.join(self.cfg.work_dir, self.output_folder),
                filename="results_spotting.json")
    
    def predict_step(self, batch, batch_idx):
        if not self.stop_predict:
            if self.runner == "runner_pooling":
                game_ID, feat_half1, feat_half2, label_half1, label_half2 = batch

                game_ID = game_ID[0]
                feat_half1 = feat_half1.squeeze(0)
                feat_half2 = feat_half2.squeeze(0)

                # Compute the output for batches of frames
                BS = 256
                timestamp_long_half_1 = timestamp_half(self.model,feat_half1,BS)
                timestamp_long_half_2 = timestamp_half(self.model,feat_half2,BS)
                
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
                            
                            json_data["predictions"].append(get_prediction_data(False,frame_index,framerate,half=half,version=self.trainer.predict_dataloaders.dataset.version,l=l,confidence=confidence, runner=self.runner))
                
                    json_data["predictions"] = sorted(json_data["predictions"], key=lambda x: int(x['position']))
                    json_data["predictions"] = sorted(json_data["predictions"], key=lambda x: int(x['half']))

                os.makedirs(os.path.join(self.cfg.work_dir, self.output_folder, game_ID), exist_ok=True)
                with open(os.path.join(self.cfg.work_dir, self.output_folder, game_ID, "results_spotting.json"), 'w') as output_file:
                    json.dump(json_data, output_file, indent=4)
            elif self.runner == "runner_JSON":
                video, features, labels = batch

                video = video[0]
                video, _ = os.path.splitext(video)
                features = features.squeeze(0)

                # Compute the output for batches of frames
                BS = 256
                timestamp_long = timestamp_half(self.model,features,BS)
                
                timestamp_long = timestamp_long[:, 1:]

                self.spotting_predictions.append(timestamp_long)

                framerate = self.trainer.predict_dataloaders.dataset.framerate
                get_spot = get_spot_from_NMS

                json_data = get_json_data(False,game_ID=video)

                # for half, timestamp in enumerate([timestamp_long_half_1, timestamp_long_half_2]):
                for l in range(self.trainer.predict_dataloaders.dataset.num_classes):
                    spots = get_spot(
                        timestamp_long[:, l], window=self.cfg.model.post_proc.NMS_window*self.cfg.model.backbone.framerate, thresh=self.cfg.model.post_proc.NMS_threshold)
                    for spot in spots:
                        # print("spot", int(spot[0]), spot[1], spot)
                        frame_index = int(spot[0])
                        confidence = spot[1]
                        if confidence < self.confidence_threshold:
                            continue
                        
                        json_data["predictions"].append(get_prediction_data(False,frame_index,framerate,version=2,l=l,confidence=confidence, runner = self.runner, inverse_event_dictionary= self.trainer.predict_dataloaders.dataset.inverse_event_dictionary))
            
                json_data["predictions"] = sorted(json_data["predictions"], key=lambda x: int(x['position']))
                # json_data["predictions"] = sorted(json_data["predictions"], key=lambda x: int(x['half']))

                os.makedirs(os.path.join(self.cfg.work_dir, self.output_folder, video), exist_ok=True)
                with open(os.path.join(self.cfg.work_dir, self.output_folder, video, "results_spotting.json"), 'w') as output_file:
                    json.dump(json_data, output_file, indent=4)

