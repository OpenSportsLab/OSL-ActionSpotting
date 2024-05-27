#!/usr/bin/env python3

import os
import copy
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from oslactionspotting.core.utils.io import load_json
from .utils import annotationstoe2eformat, get_stride, get_num_frames, read_fps
from .transform import (
    RandomGaussianNoise,
    RandomHorizontalFlipFLow,
    RandomOffsetFlow,
    SeedableRandomSquareCrop,
    ThreeCrop,
)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TARGET_HEIGHT = 224
TARGET_WIDTH = 398


class FrameReader:
    """Class used to read a video and create a clip of frames by applying some transformations.

    Args:
        modality (string): Modality of the frames.
        crop_transform : Transformations that apply to frame for cropping.
        img_transform : Transformations that apply to frame like GaussianBlur and Normalization.
        same_transform (bool): Whether to apply same trasnforms to every frame of a same clip.
        sample_fps (int): Fps at which we want to extract frames from the video.
    """

    def __init__(
        self,
        modality,
        crop_transform,
        img_transform,
        same_transform,
        sample_fps=2,
    ):
        self._is_flow = modality == "flow"
        self._crop_transform = crop_transform
        self._img_transform = img_transform
        self._same_transform = same_transform
        self._sample_fps = sample_fps

    def adapt_frame_ocv(self, frame):
        """Apply some modifications to the frame to have the expected shape and format.

        Args:
            frame (np.array).

        Returns:
            img (torch.tensor).
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(frame).float() / 255
        img = img.permute(2, 0, 1)
        if self._is_flow:
            img = img[1:, :, :]  # GB channels contain data
        return img

    def load_frames_ocv(self, video_name, start, end, pad=False):
        """Load frames from a video to create a clip of frames.

        Args:
            video_name (string): The path of the video. This is the full path of the file if we infer a single video
            or the partial path if we infer from json file.
            start (int): Start frame at which we load the clip.
            end (int): End frame at which we finish the clip.
            pad (bool): Whether to apply padding to the clip or not.
        """

        def get_stride(src_fps):
            """Get stride to apply based on the input and output fps.

            Args:
                src_fps (int): The input fps of the video.
            """
            if self._sample_fps <= 0:
                stride_extract = 1
            else:
                stride_extract = int(src_fps / self._sample_fps)
            return stride_extract

        # if self.infer:
        #     video_path = video_name
        # else:
        #     video_path = os.path.join(self._video_dir, video_name + self.extension)
        vc = cv2.VideoCapture(video_name)
        fps = vc.get(cv2.CAP_PROP_FPS)

        oh = TARGET_HEIGHT
        ow = TARGET_WIDTH

        frames = []
        rand_crop_state = None
        rand_state_backup = None
        ret = []
        n_pad_start = 0
        n_pad_end = 0
        stride_extract = get_stride(fps)
        vc.set(cv2.CAP_PROP_POS_FRAMES, start * stride_extract)
        out_frame_num = 0
        i = 0
        while True:
            ret, frame = vc.read()
            if ret:
                if i % stride_extract == 0:
                    if frame.shape[0] != oh or frame.shape[1] != ow:
                        frame = cv2.resize(frame, (ow, oh))
                    img = self.adapt_frame_ocv(frame)
                    if self._crop_transform:
                        if self._same_transform:
                            if rand_crop_state is None:
                                rand_crop_state = random.getstate()
                            else:
                                rand_state_backup = random.getstate()
                                random.setstate(rand_crop_state)

                        img = self._crop_transform(img)

                        if rand_state_backup is not None:
                            # Make sure that rand state still advances
                            random.setstate(rand_state_backup)
                            rand_state_backup = None
                    if not self._same_transform:
                        img = self._img_transform(img)
                    frames.append(img)
                    out_frame_num += 1
                i += 1
                if out_frame_num == (end - start):
                    break
            else:
                n_pad_end = (end - start) - out_frame_num
                break
        vc.release()
        # In the multicrop case, the shape is (B, T, C, H, W)
        frames = torch.stack(frames, dim=int(len(frames[0].shape) == 4))
        if self._same_transform:
            frames = self._img_transform(frames)

        # Always pad start, but only pad end if requested
        if n_pad_start > 0 or (pad and n_pad_end > 0):
            frames = nn.functional.pad(
                frames, (0, 0, 0, 0, 0, 0, n_pad_start, n_pad_end if pad else 0)
            )
        return frames


# Pad the start/end of videos with empty frames
DEFAULT_PAD_LEN = 5


def _get_deferred_rgb_transform():
    img_transforms = [
        # Jittering separately is faster (low variance)
        transforms.RandomApply(
            nn.ModuleList([transforms.ColorJitter(hue=0.2)]), p=0.25
        ),
        transforms.RandomApply(
            nn.ModuleList([transforms.ColorJitter(saturation=(0.7, 1.2))]), p=0.25
        ),
        transforms.RandomApply(
            nn.ModuleList([transforms.ColorJitter(brightness=(0.7, 1.2))]), p=0.25
        ),
        transforms.RandomApply(
            nn.ModuleList([transforms.ColorJitter(contrast=(0.7, 1.2))]), p=0.25
        ),
        # Jittering together is slower (high variance)
        # transforms.RandomApply(
        #     nn.ModuleList([
        #         transforms.ColorJitter(
        #             brightness=(0.7, 1.2), contrast=(0.7, 1.2),
        #             saturation=(0.7, 1.2), hue=0.2)
        #     ]), 0.8),
        transforms.RandomApply(nn.ModuleList([transforms.GaussianBlur(5)]), p=0.25),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return torch.jit.script(nn.Sequential(*img_transforms))


def _load_frame_deferred(gpu_transform, batch, device):
    """Load frames on the device and applying some transforms.

    Args:
        gpu_transform : Transform to apply to the frames.
        batch : Batch containing the frames and possibly some other datas as
        "mix_weight" and "mix_frame" is mixup is applied while processing videos.
        device : The device on which we load the data.

    Returns:
        frame (torch.tensor).
    """
    frame = batch["frame"].to(device)
    with torch.no_grad():
        for i in range(frame.shape[0]):
            frame[i] = gpu_transform(frame[i])

        if "mix_weight" in batch:
            weight = batch["mix_weight"].to(device)
            frame *= weight[:, None, None, None, None]

            frame_mix = batch["mix_frame"]
            for i in range(frame.shape[0]):
                frame[i] += (1.0 - weight[i]) * gpu_transform(frame_mix[i].to(device))
    return frame


def _get_img_transforms(
    is_eval, crop_dim, modality, same_transform, defer_transform=False, multi_crop=False
):
    """Get the cropping transformations and some images transformations that will be applied.

    Args:
        is_eval (bool): Whether we want train or eval transformations.
        crop_dim (int): Dimension for cropping.
        modality (string): Modality of the frame.
        same_transform (bool): Whether to apply same transform to each frame.
        defer_transform (bool): Whether some transforms have been defered to gpu.
            Default: False.
        multi_crop (bool): Whether multi cropping is applied.
            Default: False.

    Returns:
        crop_transform
        img_transform
    """
    crop_transform = None
    if crop_dim is not None:
        if multi_crop:
            assert is_eval
            crop_transform = ThreeCrop(crop_dim)
        elif is_eval:
            crop_transform = transforms.CenterCrop(crop_dim)
        elif same_transform:
            print("=> Using seeded crops!")
            crop_transform = SeedableRandomSquareCrop(crop_dim)
        else:
            crop_transform = transforms.RandomCrop(crop_dim)

    img_transforms = []
    if modality == "rgb":
        if not is_eval:
            img_transforms.append(transforms.RandomHorizontalFlip())

            if not defer_transform:
                img_transforms.extend(
                    [
                        # Jittering separately is faster (low variance)
                        transforms.RandomApply(
                            nn.ModuleList([transforms.ColorJitter(hue=0.2)]), p=0.25
                        ),
                        transforms.RandomApply(
                            nn.ModuleList(
                                [transforms.ColorJitter(saturation=(0.7, 1.2))]
                            ),
                            p=0.25,
                        ),
                        transforms.RandomApply(
                            nn.ModuleList(
                                [transforms.ColorJitter(brightness=(0.7, 1.2))]
                            ),
                            p=0.25,
                        ),
                        transforms.RandomApply(
                            nn.ModuleList(
                                [transforms.ColorJitter(contrast=(0.7, 1.2))]
                            ),
                            p=0.25,
                        ),
                        # Jittering together is slower (high variance)
                        # transforms.RandomApply(
                        #     nn.ModuleList([
                        #         transforms.ColorJitter(
                        #             brightness=(0.7, 1.2), contrast=(0.7, 1.2),
                        #             saturation=(0.7, 1.2), hue=0.2)
                        #     ]), p=0.8),
                        transforms.RandomApply(
                            nn.ModuleList([transforms.GaussianBlur(5)]), p=0.25
                        ),
                    ]
                )

        if not defer_transform:
            img_transforms.append(
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            )
    else:
        raise NotImplementedError(modality)

    img_transform = torch.jit.script(nn.Sequential(*img_transforms))
    return crop_transform, img_transform


def _print_info_helper(src_file, labels):
    """Print informations about videos contained in a json file.

    Args:
        src_file (string): The source file.
        labels (list(dict)): List containing a dict fro each video.
    """
    num_frames = sum([x["num_frames"] for x in labels])
    num_events = sum([len(x["events"]) for x in labels])
    print(
        "{} : {} videos, {} frames, {:0.5f}% non-bg".format(
            src_file, len(labels), num_frames, num_events / num_frames * 100
        )
    )


IGNORED_NOT_SHOWN_FLAG = False


class ActionSpotDataset(Dataset):
    """Class that overrides Dataset class. This class is to prepare training data using opencv.
    Training data consists of frames, associated labels and a boolean indicating if the clip of frames contains an event.
    A training sample can be mixed up with another one if mixup is used or not.
    In particular, a training sample contains the following informations without mixup:
        "frame": The frames.
        "contains_event": True if event occurs within these frames, False otherwise.
        "label": The labels associated to the frames.
    and the following informations with mixup:
        "frame": A combination of the frames of the first video and the second one.
        "contains_event": True if event occurs within these frames, False otherwise.
        "label": Rearrangement of the labels of each video.
        "mix_frame": Frames of the second video.
        "mix_weight": The weight that have been used for mixing frames and labels.

    Args:
        classes (dict): dict of class names to idx.
        label_file (list[string]|string): Path to label json files. Can be a single file or a list or a json files.
        video_dir (list[string]|string): Path to folders where videos are located. Can be a single folder or a list of folders. Must match the number of json files.
        modality (string): [rgb] Modality of the frame.
        clip_len (int): Length of a clip of frames.
        input_fps (int): Fps of the input videos.
        extract_fps (int): Fps at which we want to extract frames.
        dataset_len (int): Number of clips.
        is_eval (bool): Disable random augmentation
            Default: True.
        crop_dim (int): The dimension for cropping frames.
            Default: None.
        same_transform (bool): Apply the same random augmentation to each frame in a clip.
            Default: True.
        dilate_len (int): Dilate ground truth labels.
            Default: 0.
        mixup (bool): Whether to mixup clips of two videos or not.
            Default: False.
        pad_len (int): Number of frames to pad the start and end of videos.
            Default: DEFAULT_PAD_LEN.
        fg_upsample: Sample foreground explicitly.
            Default: -1.
    """

    def __init__(
        self,
        classes,
        label_file,
        video_dir,
        modality,
        clip_len,
        input_fps,
        extract_fps,
        dataset_len,
        is_eval=True,
        crop_dim=None,
        # stride=1,  # Downsample frame rate
        same_transform=True,
        dilate_len=0,
        mixup=False,
        pad_len=DEFAULT_PAD_LEN,
        fg_upsample=-1,
    ):
        self._src_file = label_file
        self._labels = annotationstoe2eformat(
            label_file, video_dir, input_fps, extract_fps, False
        )
        # self._labels = load_json(label_file)
        self._class_dict = classes
        self._video_idxs = {x["video"]: i for i, x in enumerate(self._labels)}
        # Sample videos weighted by their length
        num_frames = [v["num_frames"] for v in self._labels]
        self._weights_by_length = np.array(num_frames) / np.sum(num_frames)

        self._clip_len = clip_len
        assert clip_len > 0
        self._stride = 1
        assert self._stride > 0
        self._dataset_len = dataset_len
        assert dataset_len > 0
        self._pad_len = pad_len
        assert pad_len >= 0
        self._is_eval = is_eval

        # Label modifications
        self._dilate_len = dilate_len
        self._fg_upsample = fg_upsample

        # Sample based on foreground labels
        if self._fg_upsample > 0:
            self._flat_labels = []
            for i, x in enumerate(self._labels):
                for event in x["events"]:
                    if event["frame"] < x["num_frames"]:
                        self._flat_labels.append((i, event["frame"]))

        self._mixup = mixup

        # Try to do defer the latter half of the transforms to the GPU
        self._gpu_transform = None
        if not is_eval and same_transform:
            if modality == "rgb":
                print("=> Deferring some RGB transforms to the GPU!")
                self._gpu_transform = _get_deferred_rgb_transform()

        crop_transform, img_transform = _get_img_transforms(
            is_eval,
            crop_dim,
            modality,
            same_transform,
            defer_transform=self._gpu_transform is not None,
        )

        self._frame_reader = FrameReader(
            modality,
            crop_transform,
            img_transform,
            same_transform,
            extract_fps,
        )

    def load_frame_gpu(self, batch, device):
        """Load frame to gpu by appliyng some transformations or not.

        Args:
            batch: Batch containing data.
            device: The device on which to load the frames.
        """
        if self._gpu_transform is None:
            frame = batch["frame"].to(device)
        else:
            frame = _load_frame_deferred(self._gpu_transform, batch, device)
        return frame

    def _sample_uniform(self):
        """Sample video metadata and a base start index uniformly based on video lengths and weights.
        Returns:
            video_meta: metadata of a video.
            base_idx: base start index for the video processing.
        """
        video_meta = random.choices(self._labels, weights=self._weights_by_length)[0]

        video_len = video_meta["num_frames"]
        base_idx = -self._pad_len * self._stride + random.randint(
            0,
            max(0, video_len - 1 + (2 * self._pad_len - self._clip_len) * self._stride),
        )
        return video_meta, base_idx

    def _sample_foreground(self):
        """Samples video metadata and a base start index focusing on foreground labels.
        Returns:
            video_meta: metadata of a video.
            base_idx: base start index for the video processing.
        """
        video_idx, frame_idx = random.choices(self._flat_labels)[0]
        video_meta = self._labels[video_idx]
        video_len = video_meta["num_frames"]

        lower_bound = max(
            -self._pad_len * self._stride, frame_idx - self._clip_len * self._stride + 1
        )
        upper_bound = min(
            video_len - 1 + (self._pad_len - self._clip_len) * self._stride, frame_idx
        )

        base_idx = (
            random.randint(lower_bound, upper_bound)
            if upper_bound > lower_bound
            else lower_bound
        )

        assert base_idx <= frame_idx
        assert base_idx + self._clip_len > frame_idx
        return video_meta, base_idx

    def _get_one(self):
        """Get a training sample for one video."""
        if self._fg_upsample > 0 and random.random() >= self._fg_upsample:
            video_meta, base_idx = self._sample_foreground()
        else:
            video_meta, base_idx = self._sample_uniform()

        labels = np.zeros(self._clip_len, np.int64)
        for event in video_meta["events"]:
            event_frame = event["frame"]

            # Index of event in label array
            label_idx = (event_frame - base_idx) // self._stride
            if (
                label_idx >= -self._dilate_len
                and label_idx < self._clip_len + self._dilate_len
            ):
                label = self._class_dict[event["label"]]
                for i in range(
                    max(0, label_idx - self._dilate_len),
                    min(self._clip_len, label_idx + self._dilate_len + 1),
                ):
                    labels[i] = label
        frames = self._frame_reader.load_frames_ocv(
            video_meta["video"],
            base_idx,
            base_idx + self._clip_len * self._stride,
            pad=True,
        )

        return {
            "frame": frames,
            "contains_event": int(np.sum(labels) > 0),
            "label": labels,
        }

    def __getitem__(self, unused):
        """Get a training sample based on one video without mixup, two otherwise."""
        ret = self._get_one()

        if self._mixup:
            mix = self._get_one()  # Sample another clip
            l = random.betavariate(0.2, 0.2)
            label_dist = np.zeros((self._clip_len, len(self._class_dict) + 1))
            label_dist[range(self._clip_len), ret["label"]] = l
            label_dist[range(self._clip_len), mix["label"]] += 1.0 - l

            if self._gpu_transform is None:
                ret["frame"] = l * ret["frame"] + (1.0 - l) * mix["frame"]
            else:
                ret["mix_frame"] = mix["frame"]
                ret["mix_weight"] = l

            ret["contains_event"] = max(ret["contains_event"], mix["contains_event"])
            ret["label"] = label_dist

        return ret

    def __len__(self):
        return self._dataset_len

    def print_info(self):
        _print_info_helper(self._src_file, self._labels)


class DatasetVideoSharedMethods:
    def get_labels(self, video):
        """Get labels of a video.

        Args:
            video (string): Name of the video.

        Returns:
            labels (np.array): Array of length being the number of frame with elements being the index of the class.
        """
        meta = self._labels[self._video_idxs[video]]
        num_frames = meta["num_frames"]
        num_labels = num_frames // self._stride
        if num_frames % self._stride != 0:
            num_labels += 1
        labels = np.zeros(num_labels, np.int64)
        for event in meta["events"]:
            frame = event["frame"]
            if frame < num_frames:
                labels[frame // self._stride] = self._class_dict[event["label"]]
        return labels

    @property
    def augment(self):
        """Whether flip or multi cropping have been applied to frames or not."""
        return self._flip or self._multi_crop

    @property
    def videos(self):
        """Return a list containing metadatas of videos sorted by their names."""
        # return [
        #     (v['video'], v['num_frames_dali'] // self._stride,
        #      v['fps'] / self._stride) for v in self._labels]
        return sorted(
            [
                (
                    v["path"],
                    # os.path.splitext(v["path"])[0],
                    v["num_frames"] // self._stride,
                    v["fps"] / self._stride,
                )
                for v in self._labels
            ]
        )

    @property
    def labels(self):
        """Return the metadatas containing in the json file."""
        assert self._stride > 0
        if self._stride == 1:
            return self._labels
        else:
            labels = []
            for x in self._labels:
                x_copy = copy.deepcopy(x)
                x_copy["fps"] /= self._stride
                x_copy["num_frames"] //= self._stride
                for e in x_copy["events"]:
                    e["frame"] //= self._stride
                labels.append(x_copy)
            return labels

    def print_info(self):
        num_frames = sum([x["num_frames"] for x in self._labels])
        num_events = sum([len(x["events"]) for x in self._labels])
        print(
            "{} : {} videos, {} frames ({} stride), {:0.5f}% non-bg".format(
                self._src_file,
                len(self._labels),
                num_frames,
                self._stride,
                num_events / num_frames * 100,
            )
        )


class ActionSpotVideoDataset(Dataset, DatasetVideoSharedMethods):
    """Class that overrides Dataset class. This class is to prepare testing data.
    Testing data consists of frames, the name of the video and index of the first frame in the video.
    This class can process as input a json file containing metadatas of video or just a video.
    Args:
        classes (dict): dict of class names to idx.
        label_file (string): Can be path to label json or path of a video.
        video_dir (string): path to folder where videos are located.
        modality (string): [rgb] Modality of the frame.
        clip_len (int): Length of a clip of frames.
        input_fps (int): Fps of the input videos.
        extract_fps (int): Fps at which we want to extract frames.
        overlap_len (int): The number of overlapping frames between consecutive clips.
        crop_dim (int): The dimension for cropping frames.
            Default: None.
        flip (bool): Whether to flip or not the frames.
            Default: False.
        multi_crop (bool): Whether multi croping or not
            Default: False.
        skip_partial_end (bool): Whether to skip a partial number of clips at the end.
            Default: True.
        pad_len (int): Number of frames to pad the start and end of videos.
            Default: DEFAULT_PAD_LEN.
    """

    def __init__(
        self,
        classes,
        label_file,
        video_dir,
        modality,
        clip_len,
        input_fps,
        extract_fps,
        overlap_len=0,
        crop_dim=None,
        pad_len=DEFAULT_PAD_LEN,
        flip=False,
        multi_crop=False,
        skip_partial_end=True,
    ):
        self._src_file = label_file
        if label_file.endswith(".json"):
            self._labels = annotationstoe2eformat(
                label_file, video_dir, input_fps, extract_fps, False
            )
            # self._labels = load_json(label_file)
        else:
            self._labels, _ = construct_labels(label_file, extract_fps)
        # self._labels = load_json(label_file)
        self._class_dict = classes
        self._video_idxs = {x["path"]: i for i, x in enumerate(self._labels)}
        self._clip_len = clip_len
        stride = 1
        self._stride = stride
        crop_transform, img_transform = _get_img_transforms(
            is_eval=True,
            crop_dim=crop_dim,
            modality=modality,
            same_transform=True,
            multi_crop=multi_crop,
        )

        # No need to enforce same_transform since the transforms are
        # deterministic
        self._frame_reader = FrameReader(
            modality,
            crop_transform,
            img_transform,
            False,
            extract_fps,
        )

        self._flip = flip
        self._multi_crop = multi_crop

        self._clips = []
        for l in self._labels:
            has_clip = False
            for i in range(
                -pad_len * self._stride,
                max(
                    0, l["num_frames"] - (overlap_len * stride) * int(skip_partial_end)
                ),  # Need to ensure that all clips have at least one frame
                (clip_len - overlap_len) * self._stride,
            ):
                has_clip = True
                self._clips.append((l["path"], l["video"], i))
            assert has_clip, l

    def __len__(self):
        return len(self._clips)

    def __getitem__(self, idx):
        """Get a dict of metadata containing the name of the video, the index of the first frame and the clip of frame.

        Args:
            idx (int): The index of the clip in the list of clips.

        Returns:
            dict :{"video","start","frame"}.
        """
        video_path, video_name, start = self._clips[idx]
        frames = self._frame_reader.load_frames_ocv(
            video_name, start, start + self._clip_len * self._stride, pad=True
        )
        # ,stride=self._stride)

        if self._flip:
            frames = torch.stack((frames, frames.flip(-1)), dim=0)

        return {"video": video_path, "start": start // self._stride, "frame": frames}

    # def get_labels(self, video):
    #     """Get labels of a video.

    #     Args:
    #         video (string): Name of the video.

    #     Returns:
    #         labels (np.array): Array of length being the number of frame with elements being the index of the class.
    #     """
    #     meta = self._labels[self._video_idxs[video]]
    #     num_frames = meta["num_frames"]
    #     num_labels = num_frames // self._stride
    #     if num_frames % self._stride != 0:
    #         num_labels += 1
    #     labels = np.zeros(num_labels, np.int64)
    #     for event in meta["events"]:
    #         frame = event["frame"]
    #         if frame < num_frames:
    #             labels[frame // self._stride] = self._class_dict[event["label"]]
    #         else:
    #             print(
    #                 "Warning: {} >= {} is past the end {}".format(
    #                     frame, num_frames, meta["video"]
    #                 )
    #             )
    #     return labels

    # @property
    # def augment(self):
    #     """Whether flip or multi cropping have been applied to frames or not."""
    #     return self._flip or self._multi_crop

    # @property
    # def videos(self):
    #     """Return a list containing metadatas of videos sorted by their names."""
    #     return sorted(
    #         [
    #             (v["video"], v["num_frames"] // self._stride, v["fps"] / self._stride)
    #             for v in self._labels
    #         ]
    #     )

    # @property
    # def labels(self):
    #     """Return the metadatas containing in the json file."""
    #     assert self._stride > 0
    #     if self._stride == 1:
    #         return self._labels
    #     else:
    #         labels = []
    #         for x in self._labels:
    #             x_copy = copy.deepcopy(x)
    #             x_copy["fps"] /= self._stride
    #             x_copy["num_frames"] //= self._stride
    #             for e in x_copy["events"]:
    #                 e["frame"] //= self._stride
    #             labels.append(x_copy)
    #         return labels

    # def print_info(self):
    #     """Print informations about videos contained in a json file."""
    #     num_frames = sum([x["num_frames"] for x in self._labels])
    #     num_events = sum([len(x["events"]) for x in self._labels])
    #     print(
    #         "{} : {} videos, {} frames ({} stride), {:0.5f}% non-bg".format(
    #             self._src_file,
    #             len(self._labels),
    #             num_frames,
    #             self._stride,
    #             num_events / num_frames * 100,
    #         )
    #     )


from nvidia.dali import pipeline_def, backend
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import tempfile
import cupy
import math


def distribute_elements(batch_size, len_devices):
    """Return a list containing the distribution of the batch along the devices.

    Args:
        batch_size (int).
        len_device (int).

    Returns:
        distribution (list): For example if batch size is 8 and there is 4 gpus, the distribution is [2,2,2,2], meaning that each gpu will process 2 samples.
    """
    quotient, remainder = divmod(batch_size, len_devices)
    distribution = [quotient] * len_devices
    if remainder > 0:
        for i in range(len(distribution)):
            distribution[i] += 1

    return distribution


class DaliDataSet(DALIGenericIterator):
    """Class that overrides DALIGenericIterator class. This class is to prepare training data using nvidia dali.
    Training data consists of frames, associated labels and a boolean indicating if the clip of frames contains an event.
    A training sample can be mixed up with another one if mixup is used or not.
    In particular, a training sample contains the following informations without mixup:
        "frame": The frames.
        "contains_event": True if event occurs within these frames, False otherwise.
        "label": The labels associated to the frames.
    and the following informations with mixup:
        "frame": A combination of the frames of the first video and the second one.
        "contains_event": True if event occurs within these frames, False otherwise.
        "label": Rearrangement of the labels of each video.
        "mix_frame": Frames of the second video.
        "mix_weight": The weight that have been used for mixing frames and labels.

    Args:
        epochs (int): Number of training epochs.
        batch_size (int).
        output_map (List[string]): List of strings which maps consecutive outputs of DALI pipelines to user specified name. Outputs will be returned from iterator as dictionary of those names. Each name should be distinct.
        devices (list[int]): List of indexes of gpu to use.
        classes (dict): dict of class names to idx.
        label_file (list[string]|string): Paths to label jsons. Can be a single json file or a list of json files.
        clip_len (int): Length of a clip of frames.
        dataset_len (int): Number of clips.
        video_dir (list[string]|string): Paths to folder where videos are located. Can be a single folder file or a list of folders. Must match the number of json files.
        input_fps (int): Fps of the input videos.
        extract_fps (int): Fps at which we extract the frames.
        is_eval (bool): Disable random augmentation
            Default: True.
        crop_dim (int): The dimension for cropping frames.
            Default: None.
        dilate_len (int): Dilate ground truth labels.
            Default: 0.
        mixup (bool): Whether to mixup clips of two videos or not.
            Default: False.
    """

    def __init__(
        self,
        epochs,
        batch_size,
        output_map,
        devices,
        classes,
        label_file,
        modality,
        clip_len,
        dataset_len,
        video_dir,
        input_fps,
        extract_fps,
        is_eval=True,
        crop_dim=None,
        dilate_len=0,
        mixup=False,
    ):
        self._src_file = label_file
        # self._labels = load_json(label_file)
        self._labels = annotationstoe2eformat(
            label_file, video_dir, input_fps, extract_fps, True
        )
        self._class_dict = classes
        self.original_batch_size = batch_size

        if mixup:
            self.batch_size = 2 * batch_size
        else:
            self.batch_size = batch_size

        self.batch_size_per_pipe = distribute_elements(self.batch_size, len(devices))

        self.batch_size = batch_size
        self.nb_videos = dataset_len * 2 if mixup else dataset_len
        self.mixup = mixup
        self.output_map = output_map
        self.devices = devices
        self.is_eval = is_eval
        self.crop_dim = crop_dim
        self.dilate_len = dilate_len
        self.clip_len = clip_len

        self._stride = get_stride(input_fps, extract_fps)

        if is_eval:
            nb_clips_per_video = math.ceil(dataset_len / len(self._labels)) * epochs
        else:
            nb_clips_per_video = math.ceil(dataset_len / len(self._labels)) * epochs

        if mixup:
            nb_clips_per_video = nb_clips_per_video * 2

        file_list_txt = ""
        for index, video in enumerate(self._labels):
            video_path = video["video"]
            # video_path = os.path.join(video_dir, video["video"] + extension)
            for _ in range(nb_clips_per_video):
                random_start = random.randint(1, video["num_frames"] - (clip_len + 1))
                file_list_txt += f"{video_path} {index} {random_start * self._stride} {(random_start+clip_len) * self._stride}\n"

        tf = tempfile.NamedTemporaryFile()
        tf.write(str.encode(file_list_txt))
        tf.flush()

        self.pipes = [
            self.video_pipe(
                batch_size=self.batch_size_per_pipe[index],
                sequence_length=clip_len,
                stride_dali=self._stride,
                step=-1,
                num_threads=8,
                device_id=i,
                file_list=tf.name,
                shard_id=index,
                num_shards=len(devices),
            )
            for index, i in enumerate(devices)
        ]

        for pipe in self.pipes:
            pipe.build()

        super().__init__(self.pipes, output_map, size=self.nb_videos)

        self.device = torch.device("cuda:{}".format(self.devices[1]))

        self.gpu_transform = None
        if not self.is_eval:
            self.gpu_transform = _get_deferred_rgb_transform()
            # self.gpu_transform = self.get_deferred_rgb_transform()

    def __next__(self):
        out = super().__next__()
        ret = self.getitem(out)
        if self.is_eval:
            frame = ret["frame"]
        else:
            frame = self.load_frame_deferred(self.gpu_transform, ret)
        return {"frame": frame, "label": ret["label"]}

    def delete(self):
        """Useful method to free memory used by gpu when the dataset is no longer needed."""
        for pipe in self.pipes:
            pipe.__del__()
            del pipe
        backend.ReleaseUnusedMemory()

    def get_attr(self, batch):
        """Return a dictionnary containing attributes of the batch.

        Args:
            batch (dict).

        Returns:
            dict :{"frames","contains_event","labels"}.
        """
        batch_labels = batch["label"]
        batch_images = batch["data"]
        sum_labels = torch.sum(
            batch_labels, dim=1 if len(batch_labels.shape) == 2 else 0
        )
        contains_event = (sum_labels > 0).int()
        return {
            "frame": batch_images,
            "contains_event": contains_event,
            "label": batch_labels,
        }

    def move_to_device(self, batch):
        """Move all tensors of the batch to a device. Useful since samples are handled by different gpus in a first time.

        Args:
            batch : Batch containing samples that are located on different gpus.
        """
        for key, tensor in batch.items():
            batch[key] = tensor.to(self.device)

    def getitem(self, data):
        """Construct and return a batch. Mixup clips of two videos if mixup is true.

        Args:
            data: List of samples that are located on different gpus.
        """
        nb_devices = len(self.devices)
        if nb_devices >= 2:
            ret = self.get_attr(data[0])
            mix = self.get_attr(data[1])
            self.move_to_device(ret)
            self.move_to_device(mix)

        if nb_devices >= 4:
            ret2 = self.get_attr(data[2])
            mix2 = self.get_attr(data[3])
            self.move_to_device(ret2)
            self.move_to_device(mix2)

        if self.mixup:
            if nb_devices >= 4:
                for key, tensor in ret.items():
                    ret[key] = torch.cat((tensor, ret2[key]))
                for key, tensor in mix.items():
                    mix[key] = torch.cat((tensor, mix2[key]))

            l = [random.betavariate(0.2, 0.2) for i in range(ret["frame"].shape[0])]
            l = torch.tensor(l)
            label_dist = torch.zeros(
                (ret["frame"].shape[0], self.clip_len, len(self._class_dict) + 1),
                device=self.device,
            )
            for i in range(ret["frame"].shape[0]):
                label_dist[i, range(self.clip_len), ret["label"][i]] = l[i].item()
                label_dist[i, range(self.clip_len), mix["label"][i]] += (
                    1.0 - l[i].item()
                )

            if self.gpu_transform is None:
                for i in range(ret["frame"].shape[0]):
                    ret["frame"][i] = (
                        l[i].item() * ret["frame"][i]
                        + (1.0 - l[i].item()) * mix["frame"][i]
                    )
            else:
                ret["mix_frame"] = mix["frame"]
                ret["mix_weight"] = l

            ret["contains_event"] = torch.max(
                ret["contains_event"], mix["contains_event"]
            )
            ret["label"] = label_dist
        else:
            if nb_devices >= 4:
                for key, tensor in ret.items():
                    ret[key] = torch.cat((tensor, mix[key], ret2[key], mix2[key]))
            elif nb_devices >= 2:
                for key, tensor in ret.items():
                    ret[key] = torch.cat((tensor, mix[key]))
        return ret

    def load_frame_deferred(self, gpu_transform, batch):
        """Load frames on the device and applying some transforms.

        Args:
            gpu_transform : Transform to apply to the frames.
            batch : Batch containing the frames and possibly some other datas as
            "mix_weight" and "mix_frame" is mixup is applied while processing videos.
            device : The device on which we load the data.

        Returns:
            frame (torch.tensor).
        """
        frame = batch["frame"]
        with torch.no_grad():
            for i in range(frame.shape[0]):
                frame[i] = gpu_transform(frame[i])

            if "mix_weight" in batch:
                weight = batch["mix_weight"].to(self.device)
                # weight = batch['mix_weight'].to(torch.device('cuda:0'))
                frame *= weight[:, None, None, None, None]

                frame_mix = batch["mix_frame"]
                for i in range(frame.shape[0]):
                    frame[i] += (1.0 - weight[i]) * gpu_transform(frame_mix[i])

        return frame

    @pipeline_def
    def video_pipe(
        self, file_list, sequence_length, stride_dali, step, shard_id, num_shards
    ):
        """Construct the pipeline to process a video. This pipeline process a clip with specified arguments such as stride,step and sequence length.
        The first step returns clip of frames with associated labels (index of the clip in the list of clips) and the index of the first frame.
        The second step is the cropping, mirroring (only if non eval) and normalizing the frames.
        The last step is to construct the list of labels (corresponding to events) corresponding with the extracted frames.

        Args:
            file_list (string): Path to the file with a list of <file label [start_frame [end_frame]]> values.
            sequence_length (int): Frames to load per sequence.
            stride_dali (int): Distance between consecutive frames in the sequence.
            step(int): Frame interval between each sequence.
            shard_id (int): Index of the shard to read.
            num_shards (int): Partitions the data into the specified number of parts.

        Returns:
            video (torch.tensor): The frames processed.
            label : the list of labels (corresponding to events) corresponding with the extracted frames.
        """
        video, label, frame_num = fn.readers.video(
            device="gpu",
            file_list=file_list,
            sequence_length=sequence_length,
            random_shuffle=True,
            shard_id=shard_id,
            num_shards=num_shards,
            image_type=types.RGB,
            file_list_include_preceding_frame=True,
            file_list_frame_num=True,
            enable_frame_num=True,
            stride=stride_dali,
            step=step,
            pad_sequences=True,
            skip_vfr_check=True,
        )
        if self.is_eval:
            video = fn.crop_mirror_normalize(
                video,
                dtype=types.FLOAT,
                # crop = self.crop_dim,
                crop=(self.crop_dim, self.crop_dim) if self.crop_dim != None else None,
                out_of_bounds_policy="trim_to_shape",
                output_layout="FCHW",
                mean=[IMAGENET_MEAN[i] * 255.0 for i in range(len(IMAGENET_MEAN))],
                std=[IMAGENET_STD[i] * 255.0 for i in range(len(IMAGENET_STD))],
            )
        else:
            video = fn.crop_mirror_normalize(
                video,
                dtype=types.FLOAT,
                output_layout="FCHW",
                # crop = self.crop_dim,
                crop=(self.crop_dim, self.crop_dim) if self.crop_dim != None else None,
                out_of_bounds_policy="trim_to_shape",
                # crop_w=self.crop_dim, crop_h=self.crop_dim,
                std=[255, 255, 255],
                mirror=fn.random.coin_flip(),
            )
        label = fn.python_function(
            label, frame_num, function=self.edit_labels, device="gpu"
        )
        return video, label

    def edit_labels(self, label, frame_num):
        """Construct a list having the same length as the number of frames. The elements of the list are the indexes (starting at 1) of the class where an event occurs, 0 otherwise.

        Args:
            label :index of the video to get the metadata.
            frame_num :index of start frame.

        Returns:
            labels (cupy.array): the list of labels (corresponding to events) corresponding with the extracted frames.
        """
        video_meta = self._labels[label.item()]
        base_idx = frame_num.item() // self._stride
        labels = cupy.zeros(self.clip_len, np.int64)

        for event in video_meta["events"]:
            event_frame = event["frame"]
            # Index of event in label array
            label_idx = (event_frame - base_idx) // 1
            if (
                label_idx >= self.dilate_len
                and label_idx < self.clip_len + self.dilate_len
            ):
                label = self._class_dict[event["label"]]
                for i in range(
                    max(0, label_idx - self.dilate_len),
                    min(self.clip_len, label_idx + self.dilate_len + 1),
                ):
                    labels[i] = label
        return labels

    def print_info(self):
        _print_info_helper(self._src_file, self._labels)


def get_remaining(data_len, batch_size):
    """Return the padding that ensures that all batches have an equal number of items, which is required with the pipeline to make sur that all clips are processed.
    Args:
        data_len (int): The length of dataset.
        batch_size (int).

    Returns:
        (int): The number of elements to add.
    """
    return (math.ceil(data_len / batch_size) * batch_size) - data_len


def construct_labels(path, extract_fps):
    """This method is used when the input of the dataset is a video file instead of a json file.
    It creates a pseudo json by processing the video to get metadatas.

    Args:
        path (string): The path of the video file.
        extract_fps (int): The fps at which we want to extract frames.

    Returns:
        List(dict): The pseudo json object.
        (int): stride at which we will process the video.
    """
    wanted_sample_fps = extract_fps
    vc = cv2.VideoCapture(path)
    fps = vc.get(cv2.CAP_PROP_FPS)
    num_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))

    sample_fps = read_fps(fps, wanted_sample_fps if wanted_sample_fps < fps else fps)
    num_frames_after = get_num_frames(
        num_frames, fps, wanted_sample_fps if wanted_sample_fps < fps else fps
    )

    return [
        {
            "video": path,
            "path": path,
            "num_frames": num_frames_after,
            "num_frames_base": num_frames,
            "num_events": 0,
            "events": [],
            "fps": sample_fps,
            "width": 398,
            "height": 224,
        }
    ], get_stride(fps, wanted_sample_fps if wanted_sample_fps < fps else fps)


class DaliDataSetVideo(DALIGenericIterator, DatasetVideoSharedMethods):
    """Class that overrides DALIGenericIterator class. This class is to prepare testing data using nvidia dali.
    Testing data consists of frames, the name of the video and index of the first frame in the video.
    This class can process as input a json file containing metadatas of video or just a video.

    Args:
        batch_size (int).
        output_map (List[string]): List of strings which maps consecutive outputs of DALI pipelines to user specified name. Outputs will be returned from iterator as dictionary of those names. Each name should be distinct.
        devices (list[int]): List of indexes of gpu to use.
        classes (dict): dict of class names to idx.
        label_file (string): Can be path to label json or path of a video.
        clip_len (int): Length of a clip of frames.
        video_dir (string): path to folder where videos are located.
        input_fps (int): Fps of the input videos.
        extract_fps (int): The fps at which we extract frames. This variable is used if dataset is a single video.
        overlap_len (int): The number of overlapping frames between consecutive clips.
            Default: 0.
        crop_dim (int): The dimension for cropping frames.
            Default: None.
        flip (bool): Whether to flip or not the frames.
            Default: False.
        multi_crop (bool): Whether multi croping or not
            Default: False.
    """

    def __init__(
        self,
        batch_size,
        output_map,
        devices,
        classes,
        label_file,
        modality,
        clip_len,
        video_dir,
        input_fps,
        extract_fps,
        overlap_len=0,
        crop_dim=None,
        flip=False,
        multi_crop=False,
    ):
        self._src_file = label_file
        # self.infer = False
        if label_file.endswith(".json"):
            self._labels = annotationstoe2eformat(
                label_file, video_dir, input_fps, extract_fps, True
            )
            stride_dali = get_stride(input_fps, extract_fps)
            # self._labels = load_json(label_file)
        else:
            # self.infer = True
            self._labels, stride_dali = construct_labels(label_file, extract_fps)
        # self._labels = self._labels[:3]
        self._class_dict = classes
        self._video_idxs = {x["path"]: i for i, x in enumerate(self._labels)}
        self._clip_len = clip_len
        self.crop_dim = crop_dim
        stride = 1
        self._stride = stride
        self._flip = flip
        self._multi_crop = multi_crop
        self.batch_size = batch_size // len(devices)
        self.devices = devices
        self._clips = []
        file_list_txt = ""
        cmp = 0
        for l in self._labels:
            has_clip = False
            for i in range(
                1,
                l[
                    "num_frames"
                ],  # Need to ensure that all clips have at least one frame
                (clip_len - overlap_len) * self._stride,
            ):
                if i + clip_len > l["num_frames"]:
                    end = l["num_frames_base"]
                else:
                    end = (i + clip_len) * stride_dali
                has_clip = True
                self._clips.append((l["path"], l["video"], i))
                # if self.infer:
                #     video_path = l["video"]
                # else:
                #     video_path = os.path.join(video_dir, l["video"] + extension)
                video_path = l["video"]
                file_list_txt += f"{video_path} {cmp} {i * stride_dali} {end}\n"
                # if cmp2 <5:
                #     print(file_list_txt)
                #     cmp2+=1
                cmp += 1
            last_video = l["video"]
            last_path = l["path"]
            assert has_clip, l

        x = get_remaining(len(self._clips), batch_size)
        for _ in range(x):
            self._clips.append((last_path, last_video, i))
            # if self.infer:
            #     video_path = l["video"]
            # else:
            #     video_path = os.path.join(video_dir, l["video"] + extension)
            video_path = l["video"]
            file_list_txt += f"{video_path} {cmp} {i * stride_dali} {end}\n"
            cmp += 1
        # print(file_list_txt)
        tf = tempfile.NamedTemporaryFile()
        tf.write(str.encode(file_list_txt))
        tf.flush()

        self.pipes = [
            self.video_pipe(
                batch_size=self.batch_size,
                sequence_length=clip_len,
                stride_dali=stride_dali,
                step=-1,
                num_threads=8,
                device_id=i,
                file_list=tf.name,
                shard_id=index,
                num_shards=len(devices),
            )
            for index, i in enumerate(devices)
        ]

        for pipe in self.pipes:
            pipe.build()

        size = len(self._clips)

        super().__init__(self.pipes, output_map, size=size)

    def __next__(self):
        out = super().__next__()
        video_names = []
        starts = cupy.zeros(len(self.devices) * self.batch_size, np.int64)
        cmp = 0
        for j in range(len(out)):
            for i in range(out[j]["label"].shape[0]):
                video_path, video_name, start = self._clips[out[j]["label"][i]]
                video_names.append(video_path)
                starts[cmp] = start
                cmp += 1
        return {
            "video": video_names,
            "start": torch.as_tensor(starts),
            "frame": torch.cat(
                ([data["data"].to(torch.device("cuda")) for data in out])
            ),
        }

    def delete(self):
        """Useful method to free memory used by gpu when the dataset is no longer needed."""
        for pipe in self.pipes:
            pipe.__del__()
            del pipe
        backend.ReleaseUnusedMemory()

    @pipeline_def
    def video_pipe(
        self, file_list, sequence_length, stride_dali, step, shard_id, num_shards
    ):
        """Construct the pipeline to process a video. This pipeline process a clip with specified arguments such as stride,step and sequence length.
        The first step returns clip of frames with associated labels (index of the clip in the list of clips) and the index of the first frame.
        The second step is the cropping, mirroring (only if non eval) and normalizing the frames.

        Args:
            file_list (string): Path to the file with a list of <file label [start_frame [end_frame]]> values.
            sequence_length (int): Frames to load per sequence.
            stride_dali (int): Distance between consecutive frames in the sequence.
            step(int): Frame interval between each sequence.
            shard_id (int): Index of the shard to read.
            num_shards (int): Partitions the data into the specified number of parts.

        Returns:
            video (torch.tensor): The frames processed.
            label : the index of the clip in the list of clips.
        """
        video, label = fn.readers.video(
            device="gpu",
            file_list=file_list,
            sequence_length=sequence_length,
            random_shuffle=False,
            shard_id=shard_id,
            num_shards=num_shards,
            image_type=types.RGB,
            file_list_include_preceding_frame=True,
            file_list_frame_num=True,
            stride=stride_dali,
            step=step,
            pad_sequences=True,
            skip_vfr_check=True,
        )

        # video = fn.resize(size=(224,398))
        video = fn.crop_mirror_normalize(
            video,
            dtype=types.FLOAT,
            output_layout="FCHW",
            crop=(self.crop_dim, self.crop_dim) if self.crop_dim != None else None,
            out_of_bounds_policy="trim_to_shape",
            mean=[IMAGENET_MEAN[i] * 255.0 for i in range(len(IMAGENET_MEAN))],
            std=[IMAGENET_STD[i] * 255.0 for i in range(len(IMAGENET_STD))],
        )

        return video, label
