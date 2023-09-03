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


def feats2clip(feats, stride, clip_length, padding = "replicate_last", off=0):
    if padding =="zeropad":
        print("beforepadding", feats.shape)
        pad = feats.shape[0] - int(feats.shape[0]/stride)*stride
        print("pad need to be", clip_length-pad)
        m = torch.nn.ZeroPad2d((0, 0, clip_length-pad, 0))
        feats = m(feats)
        print("afterpadding", feats.shape)
        # nn.ZeroPad2d(2)

    idx = torch.arange(start=0, end=feats.shape[0]-1, step=stride)
    idxs = []
    for i in torch.arange(-off, clip_length-off):
        idxs.append(idx+i)
    idx = torch.stack(idxs, dim=1)

    if padding=="replicate_last":
        idx = idx.clamp(0, feats.shape[0]-1)
    # print(idx)
    return feats[idx,...]


class ClipsfromJSON(Dataset):
    def __init__(self, path,
                framerate=2,
                window_size=15):
        self.path = path
        self.window_size_frame = window_size*framerate
        
        with open(path) as f :
            self.data_json = json.load(f)

        self.classes = self.data_json["labels"]
        self.num_classes = len(self.classes)

        logging.info("Pre-compute clips")

        self.features_clips = list()
        self.labels_clips = list()

        # loop over videos
        for video in tqdm(self.data_json["videos"]):
            # Load features
            features = np.load(os.path.join(os.path.dirname(path), video["path_features"]))
            features = features.reshape(-1, features.shape[-1])
            
            # convert video features into clip features
            features = feats2clip(torch.from_numpy(features), stride=self.window_size_frame, clip_length=self.window_size_frame)

            # Load labels
            labels = np.zeros((features.shape[0], self.num_classes+1))
            labels[:,0]=1 # those are BG classes

            # loop annotation for that video
            for annotation in video["annotations"]:

                time = annotation["gameTime"]
                event = annotation["label"]

                minutes = int(time[-5:-3])
                seconds = int(time[-2::])
                frame = framerate * ( seconds + 60 * minutes ) 


                if event not in self.classes:
                    continue
                label = self.classes.index(event)

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
        return self.features_clips[index,:,:], self.labels_clips[index,:]

    def __len__(self):
        return len(self.features_clips)



class VideosfromJSON(Dataset):
    def __init__(self, path,
                framerate=2,
                window_size=15):
        self.path = path
        self.window_size_frame = window_size*framerate
        self.framerate = framerate

        with open(path) as f :
            self.data_json = json.load(f)


        self.classes = self.data_json["labels"]
        self.num_classes = len(self.classes)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            features (np.array): features for the 1st half.
            labels (np.array): labels (one-hot) for the 1st half.
        """

        video = self.data_json["videos"][index]

        # Load features
        features = np.load(os.path.join(os.path.dirname(self.path), video["path_features"]))
        features = features.reshape(-1, features.shape[-1])

        # Load labels
        labels = np.zeros((features.shape[0], self.num_classes))


        for annotation in video["annotations"]:

            time = annotation["gameTime"]
            event = annotation["label"]

            half = int(time[0])

            minutes = int(time[-5:-3])
            seconds = int(time[-2::])
            frame = self.framerate * ( seconds + 60 * minutes ) 

            if event not in self.classes:
                continue
            label = self.classes.index(event)

            # if half == 1:
            frame = min(frame, features.shape[0]-1)
            labels[frame][label] = 1
               

        features = feats2clip(torch.from_numpy(features), 
                        stride=1, off=int(self.window_size_frame/2), 
                        clip_length=self.window_size_frame)

        
        return video["path_features"], features, labels

    def __len__(self):
        return len(self.data_json["videos"])

