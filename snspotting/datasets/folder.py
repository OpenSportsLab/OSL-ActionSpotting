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
# , EVENT_DICTIONARY_V2, INVERSE_EVENT_DICTIONARY_V2
# from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V1, INVERSE_EVENT_DICTIONARY_V1



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


class FolderClips(Dataset):
    def __init__(self, path,
                framerate=2,
                window_size=15):
        self.path = path
        # self.listGames = getListGames(split)
        # self.features = features
        self.window_size_frame = window_size*framerate
        # self.version = version
        


        with open(path) as f :
            self.data_json = json.load(f)

        # if version == 1:
        #     self.num_classes = 3
        #     self.labels="Labels.json"
        # elif version == 2:
        #     self.dict_event = EVENT_DICTIONARY_V2
        #     self.num_classes = 17
        #     self.labels="Labels-v2.json"
        self.classes = self.data_json["labels"]
        # print(self.dict_event)
        self.num_classes = len(self.classes)
        # print(self.num_classes)
        # logging.info("Checking/Download features and labels locally")
        # downloader = SoccerNetDownloader(path)
        # downloader.downloadGames(files=[self.labels, f"1_{self.features}", f"2_{self.features}"], split=split, verbose=False,randomized=True)


        logging.info("Pre-compute clips")

        self.game_feats = list()
        self.game_labels = list()

        # game_counter = 0
        for video in tqdm(self.data_json["videos"]):
            # Load features
            feat_half1 = np.load(os.path.join(os.path.dirname(path), video["path_features"]))
            feat_half1 = feat_half1.reshape(-1, feat_half1.shape[-1])
            # feat_half2 = np.load(os.path.join(self.path, game, "2_" + self.features))
            # feat_half2 = feat_half2.reshape(-1, feat_half2.shape[-1])

            feat_half1 = feats2clip(torch.from_numpy(feat_half1), stride=self.window_size_frame, clip_length=self.window_size_frame)
            # feat_half2 = feats2clip(torch.from_numpy(feat_half2), stride=self.window_size_frame, clip_length=self.window_size_frame)

            # Load labels
            # labels = json.load(open(os.path.join(self.path, game, self.labels)))
            labels = video["annotations"]

            label_half1 = np.zeros((feat_half1.shape[0], self.num_classes+1))
            label_half1[:,0]=1 # those are BG classes
            # label_half2 = np.zeros((feat_half2.shape[0], self.num_classes+1))
            # label_half2[:,0]=1 # those are BG classes


            for annotation in labels:

                time = annotation["gameTime"]
                event = annotation["label"]

                half = int(time[0])

                minutes = int(time[-5:-3])
                seconds = int(time[-2::])
                frame = framerate * ( seconds + 60 * minutes ) 

                # if version == 1:
                #     if "card" in event: label = 0
                #     elif "subs" in event: label = 1
                #     elif "soccer" in event: label = 2
                #     else: continue
                # elif version == 2:
                # print(event)
                # print(self.classes)
                if event not in self.classes:
                    continue
                label = self.classes.index(event)
                # label = self.classes[event]

                # if label outside temporal of view
                if frame//self.window_size_frame>=label_half1.shape[0]:
                    continue
                # if half == 2 and frame//self.window_size_frame>=label_half2.shape[0]:
                #     continue

                # if half == 1:
                label_half1[frame//self.window_size_frame][0] = 0 # not BG anymore
                label_half1[frame//self.window_size_frame][label+1] = 1 # that's my class

                # if half == 2:
                #     label_half2[frame//self.window_size_frame][0] = 0 # not BG anymore
                #     label_half2[frame//self.window_size_frame][label+1] = 1 # that's my class
            
            self.game_feats.append(feat_half1)
            # self.game_feats.append(feat_half2)
            self.game_labels.append(label_half1)
            # self.game_labels.append(label_half2)

        self.game_feats = np.concatenate(self.game_feats)
        self.game_labels = np.concatenate(self.game_labels)



    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            clip_feat (np.array): clip of features.
            clip_labels (np.array): clip of labels for the segmentation.
            clip_targets (np.array): clip of targets for the spotting.
        """
        return self.game_feats[index,:,:], self.game_labels[index,:]

    def __len__(self):
        return len(self.game_feats)


class FolderGames(Dataset):
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
            feat_half1 (np.array): features for the 1st half.
            label_half1 (np.array): labels (one-hot) for the 1st half.
        """

        video = self.data_json["videos"][index]

        # Load features
        feat_half1 = np.load(os.path.join(os.path.dirname(self.path), video["path_features"]))
        feat_half1 = feat_half1.reshape(-1, feat_half1.shape[-1])

        # Load labels
        label_half1 = np.zeros((feat_half1.shape[0], self.num_classes))


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
            frame = min(frame, feat_half1.shape[0]-1)
            label_half1[frame][label] = 1

    
            

        feat_half1 = feats2clip(torch.from_numpy(feat_half1), 
                        stride=1, off=int(self.window_size_frame/2), 
                        clip_length=self.window_size_frame)


        
        return video["path_features"], feat_half1, label_half1

    def __len__(self):
        return len(self.data_json["videos"])

