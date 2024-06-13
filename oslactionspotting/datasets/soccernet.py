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
from SoccerNet.Evaluation.utils import (
    AverageMeter,
    EVENT_DICTIONARY_V2,
    INVERSE_EVENT_DICTIONARY_V2,
)
from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V1, INVERSE_EVENT_DICTIONARY_V1

from oslactionspotting.datasets.utils import (
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


class SoccerNetGame(Dataset):
    """Parent class that is used to modularise common operations between the classes that prepare features of a single game of the soccernet dataset.

    Args:
        path (string): Path of the game.
        features (str): Name of the features.
            Default: ResNET_PCA512.npy.
        version (int): Version of the dataset.
            Default: 1.
        framerate (int): The framerate at which the features have been extracted.
            Default: 2.
    """

    def __init__(self, path, features="ResNET_PCA512.npy", version=1, framerate=2):
        self.path = path
        self.framerate = framerate
        self.version = version
        self.features = features
        self.listGames = [self.path]
        if version == 1:
            self.dict_event = EVENT_DICTIONARY_V1
            self.num_classes = 3
        elif version == 2:
            self.dict_event = EVENT_DICTIONARY_V2
            self.num_classes = 17

    def __getitem__(self, index):
        pass

    def __len__(self):
        return 2

    def load_features(self):
        self.feat_half1 = np.load(os.path.join(self.path, "1_" + self.features))
        self.feat_half2 = np.load(os.path.join(self.path, "2_" + self.features))


class SoccerNetGameClips(SoccerNetGame):
    """Class that inherits from SoccerNetGame to prepare features data of a single game as clips of features.
    This class is used for the pooling methods.

    Args:
        path (string): Path of the game.
        features (str): Name of the features.
            Default: ResNET_PCA512.npy.
        version (int): Version of the dataset.
            Default: 1.
        framerate (int): The framerate at which the features have been extracted.
            Default: 2.
        window_size (int): The length of the window that will be used for the length of the clips.
            Default: 15.
    """

    def __init__(
        self, path, features="ResNET_PCA512.npy", version=1, framerate=2, window_size=15
    ):

        super().__init__(path, features, version, framerate)
        self.window_size_frame = window_size * self.framerate

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            path: path of the game.
            clip_feat_1 (np.array): clip of features for the first half.
            clip_feat_2 (np.array): clip of features for the second half.
            empty list
            empty list
        """
        # Load features
        feat_half1, feat_half2 = self.load_features()

        feat_half1 = feats2clip(
            torch.from_numpy(feat_half1),
            stride=1,
            off=int(self.window_size_frame / 2),
            clip_length=self.window_size_frame,
        )

        feat_half2 = feats2clip(
            torch.from_numpy(feat_half2),
            stride=1,
            off=int(self.window_size_frame / 2),
            clip_length=self.window_size_frame,
        )

        return self.path, feat_half1, feat_half2, [], []

    def load_features(self):
        super().load_features()
        self.feat_half1 = self.feat_half1.reshape(-1, self.feat_half1.shape[-1])
        self.feat_half2 = self.feat_half2.reshape(-1, self.feat_half2.shape[-1])
        return self.feat_half1, self.feat_half2


class SoccerNetGameClipsChunks(SoccerNetGame):
    """Class that inherits from SoccerNetGame to prepare features data of a single game as clips of features based on a chunk approach.
    This class is used for the CALF method.

    Args:
        path (string): Path of the game.
        features (str): Name of the features.
            Default: ResNET_PCA512.npy.
        version (int): Version of the dataset.
            Default: 1.
        framerate (int): The framerate at which the features have been extracted.
            Default: 2.
        chunk_size (int): Size of the chunk.
            Default: 240.
        receptive_field (int):  temporal receptive field of x seconds on both sides of the central frame in the temporal dimension of the 3D convolutions
            Default: 80.
    """

    def __init__(
        self,
        path,
        features="ResNET_PCA512.npy",
        framerate=2,
        chunk_size=240,
        receptive_field=80,
    ):
        super().__init__(path, features, 2, framerate)
        self.chunk_size = chunk_size
        self.receptive_field = receptive_field

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            clip_feat_1 (np.array): clip of features for the first half.
            clip_feat_2 (np.array): clip of features for the second half.
            clip_label_1 (np.array): clip of labels for the first half.
            clip_label_2 (np.array): clip of labels for the second half.
        """
        # Load features
        feat_half1, feat_half2 = self.load_features()

        # Load labels
        label_half1, label_half2 = self.load_labels(feat_half1, feat_half2)

        feat_half1 = feats2clip(
            torch.from_numpy(feat_half1),
            stride=self.chunk_size - self.receptive_field,
            clip_length=self.chunk_size,
            modif_last_index=True,
        )

        feat_half2 = feats2clip(
            torch.from_numpy(feat_half2),
            stride=self.chunk_size - self.receptive_field,
            clip_length=self.chunk_size,
            modif_last_index=True,
        )

        return (
            feat_half1,
            feat_half2,
            torch.from_numpy(label_half1),
            torch.from_numpy(label_half2),
        )

    def load_features(
        self,
    ):
        super().load_features()
        return self.feat_half1, self.feat_half2

    def load_labels(self, feat_half1, feat_half2):
        self.label_half1 = np.zeros((feat_half1.shape[0], self.num_classes))
        self.label_half2 = np.zeros((feat_half2.shape[0], self.num_classes))
        return self.label_half1, self.label_half2


class SoccerNet(Dataset):
    """Parent class that is used to modularise common operations between the classes that prepare features of a split of a soccernet dataset.

    Args:
        path (string): Data root of the feature files.
        features (str): Name of the features.
            Default: ResNET_PCA512.npy.
        split (List(string)): List of splits.
            Default: ["train"].
        version (int): Version of the dataset.
            Default: 1.
        framerate (int): The framerate at which the features have been extracted.
            Default: 2.
    """

    def __init__(
        self,
        path,
        features="ResNET_PCA512.npy",
        split=["train"],
        version=1,
        framerate=2,
    ):
        self.path = path
        # split=["train"] if clips else ["test"]
        self.listGames = getListGames(split)
        self.features = features
        self.framerate = framerate
        self.split = split
        self.version = version

        if version == 1:
            self.dict_event = EVENT_DICTIONARY_V1
            self.num_classes = 3
            self.labels = "Labels.json"
        elif version == 2:
            self.dict_event = EVENT_DICTIONARY_V2
            self.num_classes = 17
            self.labels = "Labels-v2.json"

    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.listGames)

    def load_features(self, index=0, game=""):
        """Load features from files.

        Args:
            index (int): Used for testing purpose to retrieve the game based on the index.
                Default: 0.
            game (string): Used for training purpose, this is the name of the game.
                Default: "".
        """
        if self.train:
            game = game
        else:
            game = self.listGames[index].replace(" ", "_")
        self.feat_half1 = np.load(os.path.join(self.path, game, "1_" + self.features))
        self.feat_half2 = np.load(os.path.join(self.path, game, "2_" + self.features))

    def load_labels(self, feat_half1, feat_half2, number_classes):
        # Load labels
        self.label_half1 = np.zeros((feat_half1.shape[0], number_classes))
        self.label_half2 = np.zeros((feat_half2.shape[0], number_classes))

    def annotation(self, annotation):
        """Process an annotation to derive the frame number, the class index and a boolean.
        Args:
            annotation (dict): It must contains the keys "gameTime" and "label".

        Returns:
            label (int): The index of the class.
            half (int): Which half of the game.
            frame (int): The index of the frame.
            cont (bool): Whether to continue in the loop or not.
        """
        time = annotation["gameTime"]
        event = annotation["label"]

        half = int(time[0])

        minutes = int(time[-5:-3])
        seconds = int(time[-2::])
        frame = self.framerate * (seconds + 60 * minutes)

        cont = False
        if self.version == 1:
            if "card" in event:
                label = 0
            elif "subs" in event:
                label = 1
            elif "soccer" in event:
                label = 2
            else:
                cont = True
        elif self.version == 2:
            if event not in self.dict_event:
                cont = True
            else:
                label = self.dict_event[event]

        return label, half, frame, cont


class SoccerNetClips(SoccerNet):
    """Class that inherits from SoccerNetGame to prepare features data of a split as clips of features.
    This class is used for the pooling methods.

    Args:
        path (string): Data root of the feature files.
        features (str): Name of the features.
            Default: ResNET_PCA512.npy.
        split (List(string)): List of splits.
            Default: ["train"].
        version (int): Version of the dataset.
            Default: 1.
        framerate (int): The framerate at which the features have been extracted.
            Default: 2.
        window_size (int): The length of the window that will be used for the length of the clips.
            Default: 15.
        train (bool): Whether training or testing.
            Default: True.
    """

    def __init__(
        self,
        path,
        features="ResNET_PCA512.npy",
        split=["train"],
        version=1,
        framerate=2,
        window_size=15,
        train=True,
    ):

        super().__init__(path, features, split, version, framerate)
        self.window_size_frame = window_size * self.framerate
        self.train = train

        logging.info("Checking/Download features and labels locally")
        downloader = SoccerNetDownloader(path)

        # if train :
        #     downloader.downloadGames(files=[self.labels, f"1_{self.features}", f"2_{self.features}"], split=split, verbose=False,randomized=True)
        # else :
        #     for s in split:
        #         if s == "challenge":
        #             downloader.downloadGames(files=[f"1_{self.features}", f"2_{self.features}"], split=[s], verbose=False,randomized=True)
        #         else:
        #             downloader.downloadGames(files=[self.labels, f"1_{self.features}", f"2_{self.features}"], split=[s], verbose=False,randomized=True)

        if train:
            logging.info("Pre-compute clips")

            self.game_feats = list()
            self.game_labels = list()

            for game in tqdm(self.listGames):
                game = game.replace(" ", "_")
                # Load features
                feat_half1, feat_half2 = self.load_features(game=game)

                feat_half1 = feats2clip(
                    torch.from_numpy(feat_half1),
                    stride=self.window_size_frame,
                    clip_length=self.window_size_frame,
                )
                feat_half2 = feats2clip(
                    torch.from_numpy(feat_half2),
                    stride=self.window_size_frame,
                    clip_length=self.window_size_frame,
                )

                # Load labels
                labels = json.load(open(os.path.join(self.path, game, self.labels)))

                label_half1, label_half2 = self.load_labels(feat_half1, feat_half2)

                for annotation in labels["annotations"]:

                    label, half, frame, cont = self.annotation(annotation)
                    if cont:
                        continue

                    # if label outside temporal of view
                    if (
                        half == 1
                        and frame // self.window_size_frame >= label_half1.shape[0]
                    ):
                        continue
                    if (
                        half == 2
                        and frame // self.window_size_frame >= label_half2.shape[0]
                    ):
                        continue

                    if half == 1:
                        label_half1[frame // self.window_size_frame][
                            0
                        ] = 0  # not BG anymore
                        label_half1[frame // self.window_size_frame][
                            label + 1
                        ] = 1  # that's my class

                    if half == 2:
                        label_half2[frame // self.window_size_frame][
                            0
                        ] = 0  # not BG anymore
                        label_half2[frame // self.window_size_frame][
                            label + 1
                        ] = 1  # that's my class
                self.game_feats.append(feat_half1)
                self.game_feats.append(feat_half2)
                self.game_labels.append(label_half1)
                self.game_labels.append(label_half2)

            self.game_feats = np.concatenate(self.game_feats)
            self.game_labels = np.concatenate(self.game_labels)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            if train :
                clip_feat (np.array): clip of features.
                clip_labels (np.array): clip of labels for the segmentation.
            if testing:
                name of the game.
                clip_feat_1 (np.array): clip of features for the first half.
                clip_feat_2 (np.array): clip of features for the second half.
                clip_label_1 (np.array): clip of labels for the first half.
                clip_label_2 (np.array): clip of labels for the second half.
        """
        if self.train:
            return self.game_feats[index, :, :], self.game_labels[index, :]
        else:
            # Load features
            feat_half1, feat_half2 = self.load_features(index=index)

            # Load labels
            label_half1, label_half2 = self.load_labels(feat_half1, feat_half2)

            # check if annotation exists
            if os.path.exists(
                os.path.join(self.path, self.listGames[index], self.labels)
            ):
                labels = json.load(
                    open(os.path.join(self.path, self.listGames[index], self.labels))
                )

                for annotation in labels["annotations"]:

                    label, half, frame, cont = self.annotation(annotation)
                    if cont:
                        continue

                    value = 1
                    if "visibility" in annotation.keys():
                        if annotation["visibility"] == "not shown":
                            value = -1

                    if half == 1:
                        frame = min(frame, feat_half1.shape[0] - 1)
                        label_half1[frame][label] = value

                    if half == 2:
                        frame = min(frame, feat_half2.shape[0] - 1)
                        label_half2[frame][label] = value

            feat_half1 = feats2clip(
                torch.from_numpy(feat_half1),
                stride=1,
                off=int(self.window_size_frame / 2),
                clip_length=self.window_size_frame,
            )

            feat_half2 = feats2clip(
                torch.from_numpy(feat_half2),
                stride=1,
                off=int(self.window_size_frame / 2),
                clip_length=self.window_size_frame,
            )

            return (
                self.listGames[index],
                feat_half1,
                feat_half2,
                label_half1,
                label_half2,
            )

    def __len__(self):
        if self.train:
            return len(self.game_feats)
        else:
            return super().__len__()

    def load_features(self, index=0, game=""):
        super().load_features(index, game)
        self.feat_half1 = self.feat_half1.reshape(-1, self.feat_half1.shape[-1])
        self.feat_half2 = self.feat_half2.reshape(-1, self.feat_half2.shape[-1])
        return self.feat_half1, self.feat_half2

    def load_labels(self, feat_half1, feat_half2):
        super().load_labels(
            feat_half1,
            feat_half2,
            self.num_classes + 1 if self.train else self.num_classes,
        )
        if self.train:
            self.label_half1[:, 0] = 1  # those are BG classes
            self.label_half2[:, 0] = 1  # those are BG classes
        return self.label_half1, self.label_half2


class SoccerNetClipsChunks(SoccerNet):
    """Class that inherits from SoccerNetGame to prepare features data of a split as clips of features with a chunk approach.
    This class is used for the CALF method.

    Args:
        path (str): Data root of the feature files.
        features (str): Name of the features.
            Default: ResNET_PCA512.npy.
        split (string): split.
            Default: "train".
        version (int): Version of the dataset.
            Default: 1.
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
        features="ResNET_PCA512.npy",
        split="train",
        framerate=2,
        chunk_size=240,
        receptive_field=80,
        chunks_per_epoch=6000,
        gpu=True,
        train=True,
    ):
        super().__init__(path, features, split, 2, framerate)
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

        logging.info("Checking/Download features and labels locally")
        downloader = SoccerNetDownloader(path)

        # if train:
        #     downloader.downloadGames(files=[self.labels, f"1_{self.features}", f"2_{self.features}"], split=split, verbose=False)
        # else:
        #     for s in split:
        #         if s == "challenge":
        #             downloader.downloadGames(files=[f"1_{self.features}", f"2_{self.features}"], split=[s], verbose=False)
        #         else:
        #             downloader.downloadGames(files=[self.labels, f"1_{self.features}", f"2_{self.features}"], split=[s], verbose=False)

        if train:
            logging.info("Pre-compute clips")

            self.game_feats = list()
            self.game_labels = list()
            self.game_anchors = list()
            for i in np.arange(self.num_classes + 1):
                self.game_anchors.append(list())

            game_counter = 0
            for game in tqdm(self.listGames):
                game = game.replace(" ", "_")
                # Load features
                feat_half1, feat_half2 = self.load_features(game=game)
                # Load labels
                labels = json.load(open(os.path.join(self.path, game, self.labels)))

                label_half1, label_half2 = self.load_labels(feat_half1, feat_half2)

                for annotation in labels["annotations"]:

                    label, half, frame, cont = self.annotation(annotation)
                    if cont:
                        continue
                    if half == 1:
                        frame = min(frame, feat_half1.shape[0] - 1)
                        label_half1[frame][label] = 1

                    if half == 2:
                        frame = min(frame, feat_half2.shape[0] - 1)
                        label_half2[frame][label] = 1

                shift_half1 = oneHotToShifts(
                    label_half1, self.K_parameters.cpu().numpy()
                )
                shift_half2 = oneHotToShifts(
                    label_half2, self.K_parameters.cpu().numpy()
                )

                anchors_half1 = getChunks_anchors(
                    shift_half1,
                    game_counter,
                    self.K_parameters.cpu().numpy(),
                    self.chunk_size,
                    self.receptive_field,
                )

                game_counter = game_counter + 1

                anchors_half2 = getChunks_anchors(
                    shift_half2,
                    game_counter,
                    self.K_parameters.cpu().numpy(),
                    self.chunk_size,
                    self.receptive_field,
                )

                game_counter = game_counter + 1
                self.game_feats.append(feat_half1)
                self.game_feats.append(feat_half2)
                self.game_labels.append(shift_half1)
                self.game_labels.append(shift_half2)

                for anchor in anchors_half1:
                    self.game_anchors[anchor[2]].append(anchor)
                for anchor in anchors_half2:
                    self.game_anchors[anchor[2]].append(anchor)

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
                clip_feat_1 (np.array): clip of features for the first half.
                clip_feat_2 (np.array): clip of features for the second half.
                clip_labels_1 (np.array): clip of labels for the first half.
                clip_labels_2 (np.array): clip of labels for the second half.

        """
        if self.train:
            # Retrieve the game index and the anchor
            class_selection = random.randint(0, self.num_classes)
            event_selection = random.randint(
                0, len(self.game_anchors[class_selection]) - 1
            )
            game_index = self.game_anchors[class_selection][event_selection][0]
            anchor = self.game_anchors[class_selection][event_selection][1]

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
            if start + self.chunk_size >= self.game_feats[game_index].shape[0]:
                start = self.game_feats[game_index].shape[0] - self.chunk_size - 1

            # Extract the clips
            clip_feat = self.game_feats[game_index][start : start + self.chunk_size]
            clip_labels = self.game_labels[game_index][start : start + self.chunk_size]

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
            # Load features
            feat_half1, feat_half2 = self.load_features(index=index)

            # Load labels
            label_half1, label_half2 = self.load_labels(feat_half1, feat_half2)

            # check if annotation exists
            if os.path.exists(
                os.path.join(
                    self.path, self.listGames[index].replace(" ", "_"), self.labels
                )
            ):
                labels = json.load(
                    open(
                        os.path.join(
                            self.path,
                            self.listGames[index].replace(" ", "_"),
                            self.labels,
                        )
                    )
                )

                for annotation in labels["annotations"]:

                    label, half, frame, cont = self.annotation(annotation)
                    if cont:
                        continue

                    value = 1
                    if "visibility" in annotation.keys():
                        if annotation["visibility"] == "not shown":
                            value = -1

                    if half == 1:
                        frame = min(frame, feat_half1.shape[0] - 1)
                        label_half1[frame][label] = value

                    if half == 2:
                        frame = min(frame, feat_half2.shape[0] - 1)
                        label_half2[frame][label] = value

            feat_half1 = feats2clip(
                torch.from_numpy(feat_half1),
                stride=self.chunk_size - self.receptive_field,
                clip_length=self.chunk_size,
                modif_last_index=True,
            )

            feat_half2 = feats2clip(
                torch.from_numpy(feat_half2),
                stride=self.chunk_size - self.receptive_field,
                clip_length=self.chunk_size,
                modif_last_index=True,
            )

            return (
                feat_half1,
                feat_half2,
                torch.from_numpy(label_half1),
                torch.from_numpy(label_half2),
            )

    def __len__(self):
        if self.train:
            return self.chunks_per_epoch
        else:
            return super().__len__()

    def load_features(self, index=0, game=""):
        super().load_features(index, game)
        return self.feat_half1, self.feat_half2

    def load_labels(self, feat_half1, feat_half2):
        super().load_labels(feat_half1, feat_half2, self.num_classes)
        return self.label_half1, self.label_half2
