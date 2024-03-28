from torch.utils.data import Dataset

import numpy as np
import random
import os
import time


from tqdm import tqdm

import torch

import logging
import json

from SoccerNet.Downloader import getListGames
from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.Evaluation.utils import AverageMeter

from snspotting.datasets.utils import getChunks_anchors, getTimestampTargets, oneHotToShifts

K_V2 = torch.FloatTensor([[-100, -98, -20, -40, -96, -5, -8, -93, -99, -31, -75, -10, -97, -75, -20, -84, -18], [-50, -49, -10, -20, -48, -3, -4, -46, -50, -15, -37, -5, -49, -38, -10, -42, -9], [50, 49, 60, 10, 48, 3, 4, 46, 50, 15, 37, 5, 49, 38, 10, 42, 9], [100, 98, 90, 20, 96, 5, 8, 93, 99, 31, 75, 10, 97, 75, 20, 84, 18]])


def feats2clip(feats, stride, clip_length, padding = "replicate_last", off=0, modif_last_index=False):
    if padding =="zeropad":
        print("beforepadding", feats.shape)
        pad = feats.shape[0] - int(feats.shape[0]/stride)*stride
        print("pad need to be", clip_length-pad)
        m = torch.nn.ZeroPad2d((0, 0, clip_length-pad, 0))
        feats = m(feats)
        print("afterpadding", feats.shape)
        # nn.ZeroPad2d(2)
    if modif_last_index: off=0
    idx = torch.arange(start=0, end=feats.shape[0]-1, step=stride)
    idxs = []
    for i in torch.arange(-off, clip_length-off):
        idxs.append(idx+i)
    idx = torch.stack(idxs, dim=1)

    if padding=="replicate_last":
        idx = idx.clamp(0, feats.shape[0]-1)
    if modif_last_index:
        idx[-1] = torch.arange(clip_length)+feats.shape[0]-clip_length
        return feats[idx,:]
    # print(idx)
    return feats[idx,...]


# class FeatureClipsfromJSON(Dataset):
#     def __init__(self, path,
#                 framerate=2,
#                 window_size=15):
#         self.path = path
#         self.window_size_frame = window_size*framerate
        
#         with open(path) as f :
#             self.data_json = json.load(f)

#         self.classes = self.data_json["labels"]
#         self.num_classes = len(self.classes)

#         logging.info("Pre-compute clips")

#         self.features_clips = list()
#         self.labels_clips = list()

#         # loop over videos
#         for video in tqdm(self.data_json["videos"]):
#             # Load features
#             features = np.load(os.path.join(os.path.dirname(path), video["path_features"].replace(' ','_')))
#             features = features.reshape(-1, features.shape[-1])
            
#             # convert video features into clip features
#             features = feats2clip(torch.from_numpy(features), stride=self.window_size_frame, clip_length=self.window_size_frame)

#             # Load labels
#             labels = np.zeros((features.shape[0], self.num_classes+1))
#             labels[:,0]=1 # those are BG classes

#             # loop annotation for that video
#             for annotation in video["annotations"]:

#                 time = annotation["gameTime"]
#                 event = annotation["label"]

#                 minutes = int(time[-5:-3])
#                 seconds = int(time[-2::])
#                 frame = framerate * ( seconds + 60 * minutes ) 


#                 if event not in self.classes:
#                     continue
#                 label = self.classes.index(event)

#                 # if label outside temporal of view
#                 if frame//self.window_size_frame>=labels.shape[0]:
#                     continue

#                 labels[frame//self.window_size_frame][0] = 0 # not BG anymore
#                 labels[frame//self.window_size_frame][label+1] = 1 # that's my class

#             self.features_clips.append(features)
#             self.labels_clips.append(labels)

#         self.features_clips = np.concatenate(self.features_clips)
#         self.labels_clips = np.concatenate(self.labels_clips)



#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index
#         Returns:
#             clip_feat (np.array): clip of features.
#             clip_labels (np.array): clip of labels.
#         """
#         return self.features_clips[index,:,:], self.labels_clips[index,:]

#     def __len__(self):
#         return len(self.features_clips)



# class FeatureVideosfromJSON(Dataset):
#     def __init__(self, path,
#                 framerate=2,
#                 window_size=15):
#         self.path = path
#         self.window_size_frame = window_size*framerate
#         self.framerate = framerate

#         with open(path) as f :
#             self.data_json = json.load(f)


#         self.classes = self.data_json["labels"]
        
#         self.num_classes = len(self.classes)

#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index
#         Returns:
#             features (np.array): features for the 1st half.
#             labels (np.array): labels (one-hot) for the 1st half.
#         """

#         video = self.data_json["videos"][index]

#         # Load features
#         features = np.load(os.path.join(os.path.dirname(self.path), video["path_features"].replace(' ','_')))
#         features = features.reshape(-1, features.shape[-1])

#         # Load labels
#         labels = np.zeros((features.shape[0], self.num_classes))


#         for annotation in video["annotations"]:

#             time = annotation["gameTime"]
#             event = annotation["label"]

#             half = int(time[0])

#             minutes = int(time[-5:-3])
#             seconds = int(time[-2::])
#             frame = self.framerate * ( seconds + 60 * minutes ) 

#             if event not in self.classes:
#                 continue
#             label = self.classes.index(event)

#             # if half == 1:
#             frame = min(frame, features.shape[0]-1)
#             labels[frame][label] = 1
               

#         features = feats2clip(torch.from_numpy(features), 
#                         stride=1, off=int(self.window_size_frame/2), 
#                         clip_length=self.window_size_frame)

        
#         return video["path_video"], features, labels

#     def __len__(self):
#         return len(self.data_json["videos"])

class FeaturefromJson(Dataset):
    def __init__(self, path, framerate=2):
        self.path = path
        self.framerate = framerate

        with open(path) as f :
            self.data_json = json.load(f)

        self.classes = self.data_json["labels"]
        self.num_classes = len(self.classes)
        self.event_dictionary = {cls: i_cls for i_cls, cls in enumerate(self.classes)}
        self.inverse_event_dictionary = {i_cls: cls for i_cls, cls in enumerate(self.classes)}
        logging.info("Pre-compute clips")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            if train :
                clip_feat (np.array): clip of features.
                clip_labels (np.array): clip of labels for the segmentation.
                clip_targets (np.array): clip of targets for the spotting.
            if games :
                feat_half (np.array): features for the half.
                label_half (np.array): labels (one-hot) for the half.
        """
        pass

    def __len__(self):
        pass
    
    def annotation(self,annotation):

        time = annotation["gameTime"]
        event = annotation["label"]
        
        minutes = int(time[-5:-3])
        seconds = int(time[-2::])
        frame = self.framerate * ( seconds + 60 * minutes )

        cont =False
        
        if event not in self.classes:
            cont=True
        else:
            label = self.classes.index(event)

        return label,frame,cont
    
class FeatureClipsfromJSON(FeaturefromJson):
    def __init__(self, path,
                framerate=2,
                window_size=15,train=True):
        super().__init__(path,framerate)
        self.window_size_frame = window_size*framerate
        self.train=train

        self.features_clips = list()
        self.labels_clips = list()

        if self.train:
            # loop over videos
            for video in tqdm(self.data_json["videos"]):
                # Load features
                features = np.load(os.path.join(os.path.dirname(path), video["path_features"].replace(' ','_')))
                features = features.reshape(-1, features.shape[-1])
                
                # convert video features into clip features
                features = feats2clip(torch.from_numpy(features), stride=self.window_size_frame, clip_length=self.window_size_frame)

                # Load labels
                labels = np.zeros((features.shape[0], self.num_classes+1))
                labels[:,0]=1 # those are BG classes

                # loop annotation for that video
                for annotation in video["annotations"]:
                    
                    label,frame,cont = self.annotation(annotation)
                    
                    if cont:
                        continue
                   
                    # if label outside temporal of view
                    if frame//self.window_size_frame>=labels.shape[0]:
                        continue

                    labels[frame//self.window_size_frame][0] = 0 # not BG anymore
                    labels[frame//self.window_size_frame][label+1] = 1 # that's my class

                self.features_clips.append(features)
                self.labels_clips.append(labels)

            self.features_clips = np.concatenate(self.features_clips)
            self.labels_clips = np.concatenate(self.labels_clips)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            clip_feat (np.array): clip of features.
            clip_labels (np.array): clip of labels.
        """
        if self.train:
            return self.features_clips[index,:,:], self.labels_clips[index,:]
        else:
            video = self.data_json["videos"][index]

            # Load features
            features = np.load(os.path.join(os.path.dirname(self.path), video["path_features"].replace(' ','_')))
            features = features.reshape(-1, features.shape[-1])

            # Load labels
            labels = np.zeros((features.shape[0], self.num_classes))


            for annotation in video["annotations"]:

                label,frame,cont = self.annotation(annotation)
                
                if cont:
                    continue
                
                frame = min(frame, features.shape[0]-1)
                labels[frame][label] = 1
                

            features = feats2clip(torch.from_numpy(features), 
                            stride=1, off=int(self.window_size_frame/2), 
                            clip_length=self.window_size_frame)

            return video["path_video"], features, labels
    def __len__(self):
        if self.train:
            return len(self.features_clips)
        else:
            return len(self.data_json["videos"])
        
class FeatureClipChunksfromJson(FeaturefromJson):
    def __init__(self, path,
                framerate=2,
                chunk_size=240, receptive_field=80, chunks_per_epoch=6000,gpu=True, train=True):
        super().__init__(path,framerate)
        self.chunk_size=chunk_size
        self.receptive_field=receptive_field
        self.chunks_per_epoch=chunks_per_epoch
        self.gpu = gpu
        self.train=train
        global K_V2
        if self.gpu >=0 :
            K_V2=K_V2.cuda()
        self.K_parameters = K_V2*framerate 
        self.num_detections =15

        if self.train:
            self.features_clips = list()
            self.labels_clips = list()
            self.anchors_clips = list()
            for i in np.arange(self.num_classes+1):
                self.anchors_clips.append(list())

            game_counter = 0
            # loop over videos
            for video in tqdm(self.data_json["videos"]):
                # Load features
                features = np.load(os.path.join(os.path.dirname(path), video["path_features"].replace(' ','_')))
                
                # Load labels
                labels = np.zeros((features.shape[0], self.num_classes))

                # loop annotation for that video
                for annotation in video["annotations"]:
                    
                    label,frame,cont = self.annotation(annotation)
                    
                    if cont:
                        continue
                    
                    frame = min(frame, features.shape[0]-1)
                    labels[frame][label] = 1
                
                shift_half = oneHotToShifts(labels, self.K_parameters.cpu().numpy())

                anchors_half = getChunks_anchors(shift_half, game_counter, self.K_parameters.cpu().numpy(), self.chunk_size, self.receptive_field)

                game_counter = game_counter+1

                self.features_clips.append(features)
                self.labels_clips.append(shift_half)

                for anchor in anchors_half:
                    self.anchors_clips[anchor[2]].append(anchor)
    
    def __getitem__(self, index):
        if self.train:
            # Retrieve the game index and the anchor
            class_selection = random.randint(0, self.num_classes)
            event_selection = random.randint(0, len(self.anchors_clips[class_selection])-1)
            game_index = self.anchors_clips[class_selection][event_selection][0]
            anchor = self.anchors_clips[class_selection][event_selection][1]

            # Compute the shift for event chunks
            if class_selection < self.num_classes:
                shift = np.random.randint(-self.chunk_size+self.receptive_field, -self.receptive_field)
                start = anchor + shift
            # Compute the shift for non-event chunks
            else:
                start = random.randint(anchor[0], anchor[1]-self.chunk_size)
            if start < 0:
                start = 0
            if start+self.chunk_size >= self.features_clips[game_index].shape[0]:
                start = self.features_clips[game_index].shape[0]-self.chunk_size-1

            # Extract the clips
            clip_feat = self.features_clips[game_index][start:start+self.chunk_size]
            clip_labels = self.labels_clips[game_index][start:start+self.chunk_size]

            # Put loss to zero outside receptive field
            clip_labels[0:int(np.ceil(self.receptive_field/2)),:] = -1
            clip_labels[-int(np.ceil(self.receptive_field/2)):,:] = -1

            # Get the spotting target
            clip_targets = getTimestampTargets(np.array([clip_labels]), self.num_detections)[0]


            return torch.from_numpy(clip_feat), torch.from_numpy(clip_labels), torch.from_numpy(clip_targets)
        else:
            video = self.data_json["videos"][index]

            # Load features
            features = np.load(os.path.join(os.path.dirname(self.path), video["path_features"].replace(' ','_')))

            # Load labels
            labels = np.zeros((features.shape[0], self.num_classes))
        
            for annotation in video["annotations"]:

                label,frame,cont=self.annotation(annotation)
                if cont:
                    continue

                value = 1
                if "visibility" in annotation.keys():
                    if annotation["visibility"] == "not shown":
                        value = -1

                frame = min(frame, features.shape[0]-1)
                labels[frame][label] = value

            features = feats2clip(torch.from_numpy(features), 
                            stride=self.chunk_size-self.receptive_field, 
                            clip_length=self.chunk_size, modif_last_index=True)

            return features, torch.from_numpy(labels)
        
    def __len__(self):
        if self.train : return self.chunks_per_epoch
        else : return len(self.data_json["videos"])