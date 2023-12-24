from torch.utils.data import Dataset

import numpy as np
import random
import os


from tqdm import tqdm

import torch

import logging
import json

from SoccerNet.Downloader import getListGames
from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.Evaluation.utils import AverageMeter, EVENT_DICTIONARY_V2, INVERSE_EVENT_DICTIONARY_V2
from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V1, INVERSE_EVENT_DICTIONARY_V1

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
    
class SoccerNet(Dataset):
    def __init__(self, path, features="ResNET_PCA512.npy", split=["train"], version=1, 
                framerate=2):
        self.path = path
        # split=["train"] if clips else ["test"]
        self.listGames = getListGames(split)
        self.features = features
        self.framerate = framerate
        self.split=split
        self.version = version


        if version == 1:
            self.dict_event = EVENT_DICTIONARY_V1
            self.num_classes = 3
            self.labels="Labels.json"
        elif version == 2:
            self.dict_event = EVENT_DICTIONARY_V2
            self.num_classes = 17
            self.labels="Labels-v2.json"

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
                feat_half1 (np.array): features for the 1st half.
                feat_half2 (np.array): features for the 2nd half.
                label_half1 (np.array): labels (one-hot) for the 1st half.
                label_half2 (np.array): labels (one-hot) for the 2nd half.
        """
        pass

    def __len__(self):
        return len(self.listGames)

    def load_features(self,index=0,game=""):
        # Load features
        self.feat_half1 = np.load(os.path.join(self.path, game if self.train else self.listGames[index], "1_" + self.features))
        self.feat_half2 = np.load(os.path.join(self.path, game if self.train else self.listGames[index], "2_" + self.features))
        
    def load_labels(self,feat_half1,feat_half2,number_classes):
        # Load labels
        self.label_half1 = np.zeros((feat_half1.shape[0], number_classes))
        self.label_half2 = np.zeros((feat_half2.shape[0], number_classes))
    
    def annotation(self,annotation):

        time = annotation["gameTime"]
        event = annotation["label"]

        half = int(time[0])

        minutes = int(time[-5:-3])
        seconds = int(time[-2::])
        frame = self.framerate * ( seconds + 60 * minutes )

        cont =False
        if self.version == 1:
            if "card" in event: label = 0
            elif "subs" in event: label = 1
            elif "soccer" in event: label = 2
            else: cont=True
        elif self.version == 2:
            if event not in self.dict_event:
                cont=True
            else:
                label = self.dict_event[event]

        return label,half,frame,cont

class SoccerNetClips(SoccerNet):
    def __init__(self,  path, features="ResNET_PCA512.npy", split=["train"], version=1, 
                framerate=2, window_size=15,train=True):
        
        super().__init__(path,features,split,version,framerate)
        self.window_size_frame = window_size*self.framerate
        self.train=train

        logging.info("Checking/Download features and labels locally")
        downloader = SoccerNetDownloader(path)

        if train : 
            downloader.downloadGames(files=[self.labels, f"1_{self.features}", f"2_{self.features}"], split=split, verbose=False,randomized=True)
        else : 
            for s in split:
                if s == "challenge":
                    downloader.downloadGames(files=[f"1_{self.features}", f"2_{self.features}"], split=[s], verbose=False,randomized=True)
                else:
                    downloader.downloadGames(files=[self.labels, f"1_{self.features}", f"2_{self.features}"], split=[s], verbose=False,randomized=True)

        if train:
            logging.info("Pre-compute clips")

            self.game_feats = list()
            self.game_labels = list()

            for game in tqdm(self.listGames):

                # Load features
                feat_half1,feat_half2=self.load_features(game=game)
                
                feat_half1 = feats2clip(torch.from_numpy(feat_half1), stride=self.window_size_frame, clip_length=self.window_size_frame)
                feat_half2 = feats2clip(torch.from_numpy(feat_half2), stride=self.window_size_frame, clip_length=self.window_size_frame)

                # Load labels
                labels = json.load(open(os.path.join(self.path, game, self.labels)))
                
                label_half1,label_half2=self.load_labels(feat_half1,feat_half2)
                
                for annotation in labels["annotations"]:

                    label,half,frame,cont=self.annotation(annotation)
                    if cont:
                        continue

                    # if label outside temporal of view
                    if half == 1 and frame//self.window_size_frame>=label_half1.shape[0]:
                        continue
                    if half == 2 and frame//self.window_size_frame>=label_half2.shape[0]:
                        continue

                    if half == 1:
                        label_half1[frame//self.window_size_frame][0] = 0 # not BG anymore
                        label_half1[frame//self.window_size_frame][label+1] = 1 # that's my class

                    if half == 2:
                        label_half2[frame//self.window_size_frame][0] = 0 # not BG anymore
                        label_half2[frame//self.window_size_frame][label+1] = 1 # that's my class
                self.game_feats.append(feat_half1)
                self.game_feats.append(feat_half2)
                self.game_labels.append(label_half1)
                self.game_labels.append(label_half2)

            self.game_feats = np.concatenate(self.game_feats)
            self.game_labels = np.concatenate(self.game_labels)

    def __getitem__(self,index):
        """
        Args:
            index (int): Index
        Returns:
            if train :
                clip_feat (np.array): clip of features.
                clip_labels (np.array): clip of labels for the segmentation.
                clip_targets (np.array): clip of targets for the spotting.
        """
        if self.train:
            return self.game_feats[index,:,:], self.game_labels[index,:]
        else:
            # Load features
            feat_half1,feat_half2=self.load_features(index=index)

            # Load labels
            label_half1,label_half2=self.load_labels(feat_half1,feat_half2)

            # check if annotation exists
            if os.path.exists(os.path.join(self.path, self.listGames[index], self.labels)):
                labels = json.load(open(os.path.join(self.path, self.listGames[index], self.labels)))

                for annotation in labels["annotations"]:

                    label,half,frame,cont=self.annotation(annotation)
                    if cont:
                        continue

                    value = 1
                    if "visibility" in annotation.keys():
                        if annotation["visibility"] == "not shown":
                            value = -1


                    if half == 1:
                        frame = min(frame, feat_half1.shape[0]-1)
                        label_half1[frame][label] = value

                    if half == 2:
                        frame = min(frame, feat_half2.shape[0]-1)
                        label_half2[frame][label] = value

            feat_half1 = feats2clip(torch.from_numpy(feat_half1), 
                            stride=1, off=int(self.window_size_frame/2), 
                            clip_length=self.window_size_frame)

            feat_half2 = feats2clip(torch.from_numpy(feat_half2), 
                            stride=1, off=int(self.window_size_frame/2), 
                            clip_length=self.window_size_frame)

            return self.listGames[index], feat_half1, feat_half2, label_half1, label_half2
            
    def __len__(self):
        if self.train :
            return len(self.game_feats)
        else:
            return super().__len__() 
        
    def load_features(self,index=0,game=""):
        super().load_features(index,game)
        self.feat_half1 = self.feat_half1.reshape(-1, self.feat_half1.shape[-1])
        self.feat_half2 = self.feat_half2.reshape(-1, self.feat_half2.shape[-1])
        return self.feat_half1,self.feat_half2

    def load_labels(self,feat_half1,feat_half2):
        super().load_labels(feat_half1,feat_half2,self.num_classes + 1 if self.train else self.num_classes)
        if self.train : 
            self.label_half1[:,0]=1 # those are BG classes
            self.label_half2[:,0]=1 # those are BG classes
        return self.label_half1,self.label_half2

class SoccerNetClipsChunks(SoccerNet):
    def __init__(self,path, features="ResNET_PCA512.npy", split="train", 
                framerate=2, chunk_size=240, receptive_field=80, chunks_per_epoch=6000,gpu=True,train=True):
        super().__init__(path,features,split,2,framerate)
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

        logging.info("Checking/Download features and labels locally")
        downloader = SoccerNetDownloader(path)

        if train:
            downloader.downloadGames(files=[self.labels, f"1_{self.features}", f"2_{self.features}"], split=split, verbose=False)
        else:
            for s in split:
                if s == "challenge":
                    downloader.downloadGames(files=[f"1_{self.features}", f"2_{self.features}"], split=[s], verbose=False)
                else:
                    downloader.downloadGames(files=[self.labels, f"1_{self.features}", f"2_{self.features}"], split=[s], verbose=False)

        if train:
            logging.info("Pre-compute clips")

            self.game_feats = list()
            self.game_labels = list()
            self.game_anchors = list()
            for i in np.arange(self.num_classes+1):
                self.game_anchors.append(list())

            game_counter = 0
            for game in tqdm(self.listGames):
                #Load features
                feat_half1,feat_half2=self.load_features(game=game)
                # Load labels
                labels = json.load(open(os.path.join(self.path, game, self.labels)))
                
                label_half1,label_half2=self.load_labels(feat_half1,feat_half2)
                
                for annotation in labels["annotations"]:

                    label,half,frame,cont=self.annotation(annotation)
                    if cont:
                        continue
                    if half == 1:
                        frame = min(frame, feat_half1.shape[0]-1)
                        label_half1[frame][label] = 1

                    if half == 2:
                        frame = min(frame, feat_half2.shape[0]-1)
                        label_half2[frame][label] = 1

                shift_half1 = oneHotToShifts(label_half1, self.K_parameters.cpu().numpy())
                shift_half2 = oneHotToShifts(label_half2, self.K_parameters.cpu().numpy())

                anchors_half1 = getChunks_anchors(shift_half1, game_counter, self.K_parameters.cpu().numpy(), self.chunk_size, self.receptive_field)

                game_counter = game_counter+1

                anchors_half2 = getChunks_anchors(shift_half2, game_counter, self.K_parameters.cpu().numpy(), self.chunk_size, self.receptive_field)

                game_counter = game_counter+1
                self.game_feats.append(feat_half1)
                self.game_feats.append(feat_half2)
                self.game_labels.append(shift_half1)
                self.game_labels.append(shift_half2)

                for anchor in anchors_half1:
                    self.game_anchors[anchor[2]].append(anchor)
                for anchor in anchors_half2:
                    self.game_anchors[anchor[2]].append(anchor)
    def __getitem__(self, index):
        if self.train:
        # Retrieve the game index and the anchor
            class_selection = random.randint(0, self.num_classes)
            event_selection = random.randint(0, len(self.game_anchors[class_selection])-1)
            game_index = self.game_anchors[class_selection][event_selection][0]
            anchor = self.game_anchors[class_selection][event_selection][1]

            # Compute the shift for event chunks
            if class_selection < self.num_classes:
                shift = np.random.randint(-self.chunk_size+self.receptive_field, -self.receptive_field)
                start = anchor + shift
            # Compute the shift for non-event chunks
            else:
                start = random.randint(anchor[0], anchor[1]-self.chunk_size)
            if start < 0:
                start = 0
            if start+self.chunk_size >= self.game_feats[game_index].shape[0]:
                start = self.game_feats[game_index].shape[0]-self.chunk_size-1

            # Extract the clips
            clip_feat = self.game_feats[game_index][start:start+self.chunk_size]
            clip_labels = self.game_labels[game_index][start:start+self.chunk_size]

            # Put loss to zero outside receptive field
            clip_labels[0:int(np.ceil(self.receptive_field/2)),:] = -1
            clip_labels[-int(np.ceil(self.receptive_field/2)):,:] = -1

            # Get the spotting target
            clip_targets = getTimestampTargets(np.array([clip_labels]), self.num_detections)[0]


            return torch.from_numpy(clip_feat), torch.from_numpy(clip_labels), torch.from_numpy(clip_targets)
        else:
            # Load features
            feat_half1,feat_half2=self.load_features(index=index)

            # Load labels
            label_half1,label_half2=self.load_labels(feat_half1,feat_half2)

            # check if annotation exists
            if os.path.exists(os.path.join(self.path, self.listGames[index], self.labels)):
                labels = json.load(open(os.path.join(self.path, self.listGames[index], self.labels)))

                for annotation in labels["annotations"]:

                    label,half,frame,cont=self.annotation(annotation)
                    if cont:
                        continue

                    value = 1
                    if "visibility" in annotation.keys():
                        if annotation["visibility"] == "not shown":
                            value = -1


                    if half == 1:
                        frame = min(frame, feat_half1.shape[0]-1)
                        label_half1[frame][label] = value

                    if half == 2:
                        frame = min(frame, feat_half2.shape[0]-1)
                        label_half2[frame][label] = value

            feat_half1 = feats2clip(torch.from_numpy(feat_half1), 
                            stride=self.chunk_size-self.receptive_field, 
                            clip_length=self.chunk_size, modif_last_index=True)

            feat_half2 = feats2clip(torch.from_numpy(feat_half2), 
                            stride=self.chunk_size-self.receptive_field, 
                            clip_length=self.chunk_size, modif_last_index=True)

            return feat_half1, feat_half2, torch.from_numpy(label_half1), torch.from_numpy(label_half2)

    def __len__(self):
        if self.train : return self.chunks_per_epoch
        else : return super().__len__()
    
    def load_features(self, index=0, game=""):
        super().load_features(index, game)
        return self.feat_half1,self.feat_half2
    
    def load_labels(self, feat_half1, feat_half2):
        super().load_labels(feat_half1, feat_half2, self.num_classes)
        return self.label_half1,self.label_half2

def rulesToCombineShifts(shift_from_last_event, shift_until_next_event, params):
    
    s1  = shift_from_last_event
    s2  = shift_until_next_event
    K = params
    
    if s1 < K[2]:
        value = s1
    elif s1 < K[3]:
        if s2 <= K[0]:
            value = s1
        else:
            if (s1-K[2])/(K[3]-K[2]) < (K[1]-s2)/(K[1]-K[0]):
                value = s1
            else:
                value = s2
    else:
        value = s2
        
    return value

def oneHotToShifts(onehot, params):

    
    nb_frames = onehot.shape[0]
    nb_actions = onehot.shape[1]
    
    Shifts = np.empty(onehot.shape)
    
    for i in range(nb_actions):
        
        x = onehot[:,i]
        K = params[:,i]
        shifts = np.empty(nb_frames)
        
        loc_events = np.where(x == 1)[0]
        nb_events = len(loc_events)
        
        if nb_events == 0:
            shifts = np.full(nb_frames, K[0])
        elif nb_events == 1:
            shifts = np.arange(nb_frames) - loc_events
        else:
            loc_events = np.concatenate(([-K[3]],loc_events,[nb_frames-K[0]]))
            for j in range(nb_frames):
                shift_from_last_event = j - loc_events[np.where(j >= loc_events)[0][-1]]
                shift_until_next_event = j - loc_events[np.where(j < loc_events)[0][0]]
                shifts[j] = rulesToCombineShifts(shift_from_last_event, shift_until_next_event, K)
        
        Shifts[:,i] = shifts
    
    return Shifts

import random


def getNegativeIndexes(labels, params, chunk_size):

    zero_one_labels = np.zeros(labels.shape)
    for i in np.arange(labels.shape[1]):
        zero_one_labels[:,i] = 1-np.logical_or(np.where(labels[:,i] >= params[3,i], 1,0),np.where(labels[:,i] <= params[0,i], 1,0))
    zero_one = np.where(np.sum(zero_one_labels, axis=1)>0, 0, 1)

    zero_one_pad = np.append(np.append([1-zero_one[0],], zero_one, axis=0), [1-zero_one[-1]], axis=0)
    zero_one_pad_shift = np.append(zero_one_pad[1:], zero_one_pad[-1])

    zero_one_sub = zero_one_pad - zero_one_pad_shift

    zero_to_one_index = np.where(zero_one_sub == -1)[0]
    one_to_zero_index = np.where(zero_one_sub == 1)[0]


    if zero_to_one_index[0] > one_to_zero_index[0]:
        one_to_zero_index = one_to_zero_index[1:]
    if zero_to_one_index.shape[0] > one_to_zero_index.shape[0]:
        zero_to_one_index = zero_to_one_index[:-1]

    list_indexes = list()

    for i,j in zip(zero_to_one_index, one_to_zero_index):
        if j-i >= chunk_size: 
            list_indexes.append([i,j])

    return list_indexes


def getChunks_anchors(labels, game_index, params, chunk_size=240, receptive_field=80):

    # get indexes of labels
    indexes=list()
    for i in np.arange(labels.shape[1]):
        indexes.append(np.where(labels[:,i] == 0)[0].tolist())

    # Positive chunks
    anchors = list()

    class_counter = 0
    for event in indexes:
        for element in event:
            anchors.append([game_index,element,class_counter])
        class_counter += 1


    # Negative chunks

    negative_indexes = getNegativeIndexes(labels, params, chunk_size)

    for negative_index in negative_indexes:
        start = [negative_index[0], negative_index[1]]
        anchors.append([game_index,start,labels.shape[1]])


    return anchors

def getTimestampTargets(labels, num_detections):

    targets = np.zeros((labels.shape[0],num_detections,2+labels.shape[-1]), dtype='float')

    for i in np.arange(labels.shape[0]):

        time_indexes, class_values = np.where(labels[i]==0)

        counter = 0

        for time_index, class_value in zip(time_indexes, class_values):

            # Confidence
            targets[i,counter,0] = 1.0 
            # frame index normalized
            targets[i,counter,1] = time_index/(labels.shape[1])
            # The class one hot encoded
            targets[i,counter,2+class_value] = 1.0
            counter += 1

            if counter >= num_detections:
                print("More timestamp than what was fixed... A lot happened in that chunk")
                break

    return targets



# Function to transform the timestamps to vectors
def timestamps2long(output_spotting, video_size, chunk_size, receptive_field):

    start = 0
    last = False
    receptive_field = receptive_field//2

    timestamps_long = torch.zeros([video_size,output_spotting.size()[-1]-2], dtype = torch.float, device=output_spotting.device)-1


    for batch in np.arange(output_spotting.size()[0]):

        tmp_timestamps = torch.zeros([chunk_size,output_spotting.size()[-1]-2], dtype = torch.float, device=output_spotting.device)-1
        
        for i in np.arange(output_spotting.size()[1]):
            tmp_timestamps[torch.floor(output_spotting[batch,i,1]*(chunk_size-1)).type(torch.int) , torch.argmax(output_spotting[batch,i,2:]).type(torch.int) ] = output_spotting[batch,i,0]

        # ------------------------------------------
        # Store the result of the chunk in the video
        # ------------------------------------------

        # For the first chunk
        if start == 0:
            timestamps_long[0:chunk_size-receptive_field] = tmp_timestamps[0:chunk_size-receptive_field]

        # For the last chunk
        elif last:
            timestamps_long[start+receptive_field:start+chunk_size] = tmp_timestamps[receptive_field:]
            break

        # For every other chunk
        else:
            timestamps_long[start+receptive_field:start+chunk_size-receptive_field] = tmp_timestamps[receptive_field:chunk_size-receptive_field]
        
        # ---------------
        # Loop Management
        # ---------------

        # Update the index
        start += chunk_size - 2 * receptive_field
        # Check if we are at the last index of the game
        if start + chunk_size >= video_size:
            start = video_size - chunk_size 
            last = True
    return timestamps_long

# Function to transform the batches to vectors
def batch2long(output_segmentation, video_size, chunk_size, receptive_field):

    start = 0
    last = False
    receptive_field = receptive_field//2

    segmentation_long = torch.zeros([video_size,output_segmentation.size()[-1]], dtype = torch.float, device=output_segmentation.device)


    for batch in np.arange(output_segmentation.size()[0]):

        tmp_segmentation = torch.nn.functional.one_hot(torch.argmax(output_segmentation[batch], dim=-1), num_classes=output_segmentation.size()[-1])


        # ------------------------------------------
        # Store the result of the chunk in the video
        # ------------------------------------------

        # For the first chunk
        if start == 0:
            segmentation_long[0:chunk_size-receptive_field] = tmp_segmentation[0:chunk_size-receptive_field]

        # For the last chunk
        elif last:
            segmentation_long[start+receptive_field:start+chunk_size] = tmp_segmentation[receptive_field:]
            break

        # For every other chunk
        else:
            segmentation_long[start+receptive_field:start+chunk_size-receptive_field] = tmp_segmentation[receptive_field:chunk_size-receptive_field]
        
        # ---------------
        # Loop Management
        # ---------------

        # Update the index
        start += chunk_size - 2 * receptive_field
        # Check if we are at the last index of the game
        if start + chunk_size >= video_size:
            start = video_size - chunk_size 
            last = True
    return segmentation_long