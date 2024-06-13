from torch.utils.data import Dataset

import numpy as np
import random
import os


from tqdm import tqdm

import torch

import logging
import json




from .utils import (
    feats2clip,
    getChunks_anchors,
    getTimestampTargets,
    oneHotToShifts,
)

K_V2 = torch.FloatTensor(
    [
        [
            -100,
            -98,
            -20,
            -40,
            -96,
            -5,
            -8,
            -93,
            -99,
            -31,
            -75,
            -10,
            -97,
            -75,
            -20,
            -84,
            -18,
        ],
        [
            -50,
            -49,
            -10,
            -20,
            -48,
            -3,
            -4,
            -46,
            -50,
            -15,
            -37,
            -5,
            -49,
            -38,
            -10,
            -42,
            -9,
        ],
        [50, 49, 60, 10, 48, 3, 4, 46, 50, 15, 37, 5, 49, 38, 10, 42, 9],
        [100, 98, 90, 20, 96, 5, 8, 93, 99, 31, 75, 10, 97, 75, 20, 84, 18],
    ]
)


class FeaturefromJson(Dataset):
    """Parent class that is used to modularise common operations between the classes that prepare features data.
    In particular, this class loads the input, creates dictionnaries of classes and has a method to process the annotations.

    Args:
        path (str|List(str)): Path of the input. Can be a json file, a features file or a list of json files (list only for training purposes).
        features_dir (str|List(str)): Path where the features are located. Must match the number of input paths.
        classes (path): Path of the file containing the classes. Only used if input is feature file, otherwise list of classes is in the json file.
        framerate (int): The framerate at which the features have been extracted.
            Default: 2.
    """

    def __init__(self, path, features_dir, classes, framerate=2):
        self.classes = classes
        self.path = path
        self.framerate = framerate
        self.features_dir = features_dir

        self.is_json = True

        if isinstance(path, list):
            self.data_json = []
            self.classes = []
            for single_path in path:
                assert os.path.isfile(single_path)
                with open(single_path) as f:
                    tmp = json.load(f)
                    self.data_json.append(tmp)
                    self.classes.append(tmp["labels"])
            assert all(x == self.classes[0] for x in self.classes) == True
            self.classes = self.classes[0]

        else:
            self.features_dir = [features_dir]

            assert os.path.isfile(path)
            if path.endswith(".json"):
                with open(path) as f:
                    tmp = json.load(f)
                    self.data_json = [tmp]
                self.classes = tmp["labels"]
            else:
                self.is_json = False
                self.data_json = [
                    {
                        "videos": [
                            {
                                "path": path,
                                "annotations": [],
                            }
                        ]
                    }
                ]
                assert isinstance(self.classes, list) or os.path.isfile(self.classes)

                from oslactionspotting.core.utils.io import load_text
                if not isinstance(self.classes, list):
                    self.classes = load_text(self.classes)

        self.num_classes = len(self.classes)
        self.event_dictionary = {cls: i_cls for i_cls, cls in enumerate(self.classes)}
        self.inverse_event_dictionary = {
            i_cls: cls for i_cls, cls in enumerate(self.classes)
        }
        logging.info("Pre-compute clips")

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

    def annotation(self, annotation):
        """Process an annotation to derive the frame number, the class index and a boolean.
        Args:
            annotation (dict): It must contains the keys "gameTime" and "label".

        Returns:
            label (int): The index of the class.
            frame (int): The index of the frame.
            cont (bool): Whether to continue in the loop or not.
        """
        # time = annotation["gameTime"]
        event = annotation["label"]

        if "position" in annotation.keys():
            frame = int(self.framerate * (int(annotation["position"]) / 1000))
        else:
            time = annotation["gameTime"]

            minutes = int(time[-5:-3])
            seconds = int(time[-2::])
            frame = self.framerate * (seconds + 60 * minutes)

        cont = False

        if event not in self.classes:
            cont = True
        else:
            label = self.classes.index(event)

        return label, frame, cont


class FeatureClipsfromJSON(FeaturefromJson):
    """Class that inherits from FeaturefromJson to prepare features data as clips of features.
    This class is used for the pooling methods.
    The class has 2 behaviours for processing the data depending if it is for training or testing purposes.

    Args:
        path (str|List(str)): Path of the input. Can be a json file, a features file or a list of json files (list only for training purposes).
        features_dir (str|List(str)): Path where the features are located. Must match the number of input paths.
        classes (path): Path of the file containing the classes. Only used if input is feature file, otherwise list of classes is in the json file.
        framerate (int): The framerate at which the features have been extracted.
            Default: 2.
        window_size (int): The length of the window that will be used for the length of the clips.
            Default: 15.
        train (bool): Whether we prepare data for training or for testing purposes.
            Default: True.
    """

    def __init__(
        self, path, features_dir, classes, framerate=2, window_size=15, train=True
    ):
        super().__init__(path, features_dir, classes, framerate)
        self.window_size_frame = window_size * framerate
        self.train = train
        self.features_clips = list()
        self.labels_clips = list()

        if self.train:
            for i, single_data_json in enumerate(self.data_json):
                if isinstance(path, list):
                    logging.info("Processing " + path[i])
                else:
                    logging.info("Processing " + path)
                # loop over videos
                for video in tqdm(single_data_json["videos"]):
                    # for video in tqdm(self.data_json["videos"]):
                    # Load features
                    features = np.load(
                        os.path.join(self.features_dir[i], video["path"])
                    )
                    features = features.reshape(-1, features.shape[-1])

                    # convert video features into clip features
                    features = feats2clip(
                        torch.from_numpy(features),
                        stride=self.window_size_frame,
                        clip_length=self.window_size_frame,
                    )

                    # Load labels
                    labels = np.zeros((features.shape[0], self.num_classes + 1))
                    labels[:, 0] = 1  # those are BG classes

                    # loop annotation for that video
                    for annotation in video["annotations"]:

                        label, frame, cont = self.annotation(annotation)

                        if cont:
                            continue

                        # if label outside temporal of view
                        if frame // self.window_size_frame >= labels.shape[0]:
                            continue

                        labels[frame // self.window_size_frame][0] = 0  # not BG anymore
                        labels[frame // self.window_size_frame][
                            label + 1
                        ] = 1  # that's my class

                    self.features_clips.append(features)
                    self.labels_clips.append(labels)

            self.features_clips = np.concatenate(self.features_clips)
            self.labels_clips = np.concatenate(self.labels_clips)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            If training:
                clip_feat (np.array): clip of features.
                clip_labels (np.array): clip of labels.
            If testing:
                Name of the feature file.
                features (np.array): clip of features.
                labels (np.array): clip of labels.
        """
        if self.train:
            return self.features_clips[index, :, :], self.labels_clips[index, :]
        else:
            video = self.data_json[0]["videos"][index]

            # Load features
            if self.is_json:
                features = np.load(os.path.join(self.features_dir[0], video["path"]))
            else:
                features = np.load(os.path.join(video["path"]))
            features = features.reshape(-1, features.shape[-1])

            # Load labels
            labels = np.zeros((features.shape[0], self.num_classes))

            if "annotations" in video.keys():
                for annotation in video["annotations"]:

                    label, frame, cont = self.annotation(annotation)

                    if cont:
                        continue

                    frame = min(frame, features.shape[0] - 1)
                    labels[frame][label] = 1

            features = feats2clip(
                torch.from_numpy(features),
                stride=1,
                off=int(self.window_size_frame / 2),
                clip_length=self.window_size_frame,
            )

            return video["path"], features, labels

    def __len__(self):
        if self.train:
            return len(self.features_clips)
        else:
            return len(self.data_json[0]["videos"])


class FeatureClipChunksfromJson(FeaturefromJson):
    """Class that inherits from FeaturefromJson to prepare features data as clips of features based on a chunk approach.
    This class is used for the CALF method.
    The class has 2 behaviours for processing the data depending if it is for training or testing purposes.

    Args:
        path (str|List(str)): Path of the input. Can be a json file, a features file or a list of json files (list only for training purposes).
        features_dir (str|List(str)): Path where the features are located. Must match the number of input paths.
        classes (path): Path of the file containing the classes. Only used if input is feature file, otherwise list of classes is in the json file.
        framerate (int): The framerate at which the features have been extracted.
            Default: 2.
        chunk_size (int): Size of the chunk.
            Default: 240.
        receptive_field (int):  temporal receptive field of x seconds on both sides of the central frame in the temporal dimension of the 3D convolutions
            Default: 80.
        chunks_per_epoch (int): Number of chunks per epoch.
            Default: 6000.
        gpu (bool): Whether gpu is used or not.
            Default: True.
        train (bool): Whether we prepare data for training or for testing purposes.
            Default: True.
    """

    def __init__(
        self,
        path,
        features_dir,
        classes,
        framerate=2,
        chunk_size=240,
        receptive_field=80,
        chunks_per_epoch=6000,
        gpu=True,
        train=True,
    ):
        super().__init__(path, features_dir, classes, framerate)
        self.chunk_size = chunk_size
        self.receptive_field = receptive_field
        self.chunks_per_epoch = chunks_per_epoch
        self.gpu = gpu
        self.train = train
        global K_V2
        if self.gpu >= 0:
            K_V2 = K_V2.cuda()
        self.K_parameters = K_V2 * framerate
        self.num_detections = 15

        if self.train:
            self.features_clips = list()
            self.labels_clips = list()
            self.anchors_clips = list()
            for i in np.arange(self.num_classes + 1):
                self.anchors_clips.append(list())

            game_counter = 0
            # loop over videos
            for i, single_data_json in enumerate(self.data_json):
                if isinstance(path, list):
                    logging.info("Processing " + path[i])
                else:
                    logging.info("Processing " + path)
                # loop over videos
                for video in tqdm(single_data_json["videos"]):
                    # for video in tqdm(self.data_json["videos"]):
                    # Load features
                    features = np.load(
                        os.path.join(self.features_dir[i], video["path"])
                    )

                    # Load labels
                    labels = np.zeros((features.shape[0], self.num_classes))

                    # loop annotation for that video
                    for annotation in video["annotations"]:

                        label, frame, cont = self.annotation(annotation)

                        if cont:
                            continue

                        frame = min(frame, features.shape[0] - 1)
                        labels[frame][label] = 1

                    shift_half = oneHotToShifts(labels, self.K_parameters.cpu().numpy())

                    anchors_half = getChunks_anchors(
                        shift_half,
                        game_counter,
                        self.K_parameters.cpu().numpy(),
                        self.chunk_size,
                        self.receptive_field,
                    )

                    game_counter = game_counter + 1

                    self.features_clips.append(features)
                    self.labels_clips.append(shift_half)

                    for anchor in anchors_half:
                        self.anchors_clips[anchor[2]].append(anchor)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            If training:
                clip_feat (np.array): clip of features.
                clip_labels (np.array): clip of labels.
                clip_targets (np.array): clip of targets.
            If testing:
                features (np.array): clip of features.
                labels (np.array): clip of labels.
        """
        if self.train:
            # Retrieve the game index and the anchor
            class_selection = random.randint(0, self.num_classes)
            event_selection = random.randint(
                0, len(self.anchors_clips[class_selection]) - 1
            )
            game_index = self.anchors_clips[class_selection][event_selection][0]
            anchor = self.anchors_clips[class_selection][event_selection][1]

            # Compute the shift for event chunks
            if class_selection < self.num_classes:
                shift = np.random.randint(
                    -self.chunk_size + self.receptive_field, -self.receptive_field
                )
                start = anchor + shift
            # Compute the shift for non-event chunks
            else:
                start = random.randint(anchor[0], anchor[1] - self.chunk_size)
            if start < 0:
                start = 0
            if start + self.chunk_size >= self.features_clips[game_index].shape[0]:
                start = self.features_clips[game_index].shape[0] - self.chunk_size - 1

            # Extract the clips
            clip_feat = self.features_clips[game_index][start : start + self.chunk_size]
            clip_labels = self.labels_clips[game_index][start : start + self.chunk_size]

            # Put loss to zero outside receptive field
            clip_labels[0 : int(np.ceil(self.receptive_field / 2)), :] = -1
            clip_labels[-int(np.ceil(self.receptive_field / 2)) :, :] = -1

            # Get the spotting target
            clip_targets = getTimestampTargets(
                np.array([clip_labels]), self.num_detections
            )[0]

            return (
                torch.from_numpy(clip_feat),
                torch.from_numpy(clip_labels),
                torch.from_numpy(clip_targets),
            )
        else:
            video = self.data_json[0]["videos"][index]

            # Load features
            if self.is_json:
                features = np.load(os.path.join(self.features_dir[0], video["path"]))
            else:
                features = np.load(os.path.join(video["path"]))

            # Load labels
            labels = np.zeros((features.shape[0], self.num_classes))

            if "annotations" in video.keys():
                for annotation in video["annotations"]:

                    label, frame, cont = self.annotation(annotation)
                    if cont:
                        continue

                    value = 1
                    if "visibility" in annotation.keys():
                        if annotation["visibility"] == "not shown":
                            value = -1

                    frame = min(frame, features.shape[0] - 1)
                    labels[frame][label] = value

            features = feats2clip(
                torch.from_numpy(features),
                stride=self.chunk_size - self.receptive_field,
                clip_length=self.chunk_size,
                modif_last_index=True,
            )

            return features, torch.from_numpy(labels)

    def __len__(self):
        if self.train:
            return self.chunks_per_epoch
        else:
            return len(self.data_json[0]["videos"])
