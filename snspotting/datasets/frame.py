#!/usr/bin/env python3

import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import time
from snspotting.core.utils.io import load_json
from .transform import RandomGaussianNoise, RandomHorizontalFlipFLow, \
    RandomOffsetFlow, SeedableRandomSquareCrop, ThreeCrop


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class FrameReader:

    IMG_NAME = '{:06d}.jpg'

    def __init__(self, frame_dir, modality, crop_transform, img_transform,
                 same_transform):
        self._frame_dir = frame_dir
        self._is_flow = modality == 'flow'
        self._crop_transform = crop_transform
        self._img_transform = img_transform
        self._same_transform = same_transform

    def read_frame(self, frame_path):
        img = torchvision.io.read_image(frame_path).float() / 255
        if self._is_flow:
            img = img[1:, :, :]     # GB channels contain data
        return img

    def load_frames(self, video_name, start, end, pad=False, stride=1,
                    randomize=False):
        rand_crop_state = None
        rand_state_backup = None
        ret = []
        n_pad_start = 0
        n_pad_end = 0
        for frame_num in range(start, end, stride):
            if randomize and stride > 1:
                frame_num += random.randint(0, stride - 1)

            if frame_num < 0:
                n_pad_start += 1
                continue

            parts = video_name.split('/')
            parts[2] = parts[2].replace('_', ' ')
            video_name = '/'.join(parts)

            frame_path = os.path.join(
                self._frame_dir, video_name,
                FrameReader.IMG_NAME.format(frame_num))
            try:
                img = self.read_frame(frame_path)
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
                ret.append(img)
                len(ret)
            except RuntimeError:
                # print('Missing file!', frame_path)
                n_pad_end += 1
        # In the multicrop case, the shape is (B, T, C, H, W)
        ret = torch.stack(ret, dim=int(len(ret[0].shape) == 4))
        
        if self._same_transform:
            ret = self._img_transform(ret)
        
        # Always pad start, but only pad end if requested
        if n_pad_start > 0 or (pad and n_pad_end > 0):
            ret = nn.functional.pad(
                ret, (0, 0, 0, 0, 0, 0, n_pad_start, n_pad_end if pad else 0))
        return ret


# Pad the start/end of videos with empty frames
DEFAULT_PAD_LEN = 5


def _get_deferred_rgb_transform():
    img_transforms = [
        # Jittering separately is faster (low variance)
        transforms.RandomApply(
            nn.ModuleList([transforms.ColorJitter(hue=0.2)]), p=0.25),
        transforms.RandomApply(
            nn.ModuleList([
                transforms.ColorJitter(saturation=(0.7, 1.2))
            ]), p=0.25),
        transforms.RandomApply(
            nn.ModuleList([
                transforms.ColorJitter(brightness=(0.7, 1.2))
            ]), p=0.25),
        transforms.RandomApply(
            nn.ModuleList([
                transforms.ColorJitter(contrast=(0.7, 1.2))
            ]), p=0.25),

        # Jittering together is slower (high variance)
        # transforms.RandomApply(
        #     nn.ModuleList([
        #         transforms.ColorJitter(
        #             brightness=(0.7, 1.2), contrast=(0.7, 1.2),
        #             saturation=(0.7, 1.2), hue=0.2)
        #     ]), 0.8),

        transforms.RandomApply(
            nn.ModuleList([transforms.GaussianBlur(5)]), p=0.25),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ]
    return torch.jit.script(nn.Sequential(*img_transforms))


def _get_deferred_bw_transform():
    img_transforms = [
        transforms.RandomApply(
            nn.ModuleList([transforms.ColorJitter(brightness=0.3)]), p=0.25),
        transforms.RandomApply(
            nn.ModuleList([transforms.ColorJitter(contrast=0.3)]), p=0.25),
        transforms.RandomApply(
            nn.ModuleList([transforms.GaussianBlur(5)]), p=0.25),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        RandomGaussianNoise()
    ]
    return torch.jit.script(nn.Sequential(*img_transforms))


def _load_frame_deferred(gpu_transform, batch, device):
    frame = batch['frame'].to(device)
    with torch.no_grad():
        for i in range(frame.shape[0]):
            frame[i] = gpu_transform(frame[i])

        if 'mix_weight' in batch:
            weight = batch['mix_weight'].to(device)
            frame *= weight[:, None, None, None, None]

            frame_mix = batch['mix_frame']
            for i in range(frame.shape[0]):
                frame[i] += (1. - weight[i]) * gpu_transform(
                    frame_mix[i].to(device))
    return frame


def _get_img_transforms(
        is_eval,
        crop_dim,
        modality,
        same_transform,
        defer_transform=False,
        multi_crop=False
):
    crop_transform = None
    if crop_dim is not None:
        if multi_crop:
            assert is_eval
            crop_transform = ThreeCrop(crop_dim)
        elif is_eval:
            crop_transform = transforms.CenterCrop(crop_dim)
        elif same_transform:
            print('=> Using seeded crops!')
            crop_transform = SeedableRandomSquareCrop(crop_dim)
        else:
            crop_transform = transforms.RandomCrop(crop_dim)

    img_transforms = []
    if modality == 'rgb':
        if not is_eval:
            img_transforms.append(
                transforms.RandomHorizontalFlip())

            if not defer_transform:
                img_transforms.extend([
                    # Jittering separately is faster (low variance)
                    transforms.RandomApply(
                        nn.ModuleList([transforms.ColorJitter(hue=0.2)]),
                        p=0.25),
                    transforms.RandomApply(
                        nn.ModuleList([
                            transforms.ColorJitter(saturation=(0.7, 1.2))
                        ]), p=0.25),
                    transforms.RandomApply(
                        nn.ModuleList([
                            transforms.ColorJitter(brightness=(0.7, 1.2))
                        ]), p=0.25),
                    transforms.RandomApply(
                        nn.ModuleList([
                            transforms.ColorJitter(contrast=(0.7, 1.2))
                        ]), p=0.25),

                    # Jittering together is slower (high variance)
                    # transforms.RandomApply(
                    #     nn.ModuleList([
                    #         transforms.ColorJitter(
                    #             brightness=(0.7, 1.2), contrast=(0.7, 1.2),
                    #             saturation=(0.7, 1.2), hue=0.2)
                    #     ]), p=0.8),

                    transforms.RandomApply(
                        nn.ModuleList([transforms.GaussianBlur(5)]), p=0.25)
                ])

        if not defer_transform:
            img_transforms.append(transforms.Normalize(
                mean=IMAGENET_MEAN, std=IMAGENET_STD))
    elif modality == 'bw':
        if not is_eval:
            img_transforms.extend([
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    nn.ModuleList([transforms.ColorJitter(hue=0.2)]), p=0.25)])
        img_transforms.append(transforms.Grayscale())

        if not defer_transform:
            if not is_eval:
                img_transforms.extend([
                    transforms.RandomApply(
                        nn.ModuleList([transforms.ColorJitter(brightness=0.3)]),
                        p=0.25),
                    transforms.RandomApply(
                        nn.ModuleList([transforms.ColorJitter(contrast=0.3)]),
                        p=0.25),
                    transforms.RandomApply(
                        nn.ModuleList([transforms.GaussianBlur(5)]), p=0.25),
                ])

            img_transforms.append(transforms.Normalize(
                mean=[0.5], std=[0.5]))

            if not is_eval:
                img_transforms.append(RandomGaussianNoise())
    elif modality == 'flow':
        assert not defer_transform

        img_transforms.append(transforms.Normalize(
            mean=[0.5, 0.5], std=[0.5, 0.5]))

        if not is_eval:
            img_transforms.extend([
                RandomHorizontalFlipFLow(),
                RandomOffsetFlow(),
                RandomGaussianNoise()
            ])
    else:
        raise NotImplementedError(modality)

    img_transform = torch.jit.script(nn.Sequential(*img_transforms))
    return crop_transform, img_transform


def _print_info_helper(src_file, labels):
        num_frames = sum([x['num_frames_2fps'] for x in labels])
        num_events = sum([len(x['events']) for x in labels])
        print('{} : {} videos, {} frames, {:0.5f}% non-bg'.format(
            src_file, len(labels), num_frames,
            num_events / num_frames * 100))


IGNORED_NOT_SHOWN_FLAG = False


class ActionSpotDataset(Dataset):

    def __init__(
            self,
            classes,                    # dict of class names to idx
            label_file,                 # path to label json
            frame_dir,                  # path to frames
            modality,                   # [rgb, bw, flow]
            clip_len,
            dataset_len,                # Number of clips
            is_eval=True,               # Disable random augmentation
            crop_dim=None,
            stride=1,                   # Downsample frame rate
            same_transform=True,        # Apply the same random augmentation to
                                        # each frame in a clip
            dilate_len=0,               # Dilate ground truth labels
            mixup=False,
            pad_len=DEFAULT_PAD_LEN,    # Number of frames to pad the start
                                        # and end of videos
            fg_upsample=-1,             # Sample foreground explicitly
    ):
        self._src_file = label_file
        self._labels = load_json(label_file)
        print("ls",len(self._labels))
        self._class_dict = classes
        self._video_idxs = {x['video']: i for i, x in enumerate(self._labels)}

        # Sample videos weighted by their length
        num_frames = [v['num_frames_2fps'] for v in self._labels]
        self._weights_by_length = np.array(num_frames) / np.sum(num_frames)

        self._clip_len = clip_len
        assert clip_len > 0
        self._stride = stride
        assert stride > 0
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
                for event in x['events']:
                    if event['frame'] < x['num_frames_2fps']:
                        self._flat_labels.append((i, event['frame']))

        self._mixup = mixup

        # Try to do defer the latter half of the transforms to the GPU
        self._gpu_transform = None
        if not is_eval and same_transform:
            if modality == 'rgb':
                print('=> Deferring some RGB transforms to the GPU!')
                self._gpu_transform = _get_deferred_rgb_transform()
            elif modality == 'bw':
                print('=> Deferring some BW transforms to the GPU!')
                self._gpu_transform = _get_deferred_bw_transform()

        crop_transform, img_transform = _get_img_transforms(
            is_eval, crop_dim, modality, same_transform,
            defer_transform=self._gpu_transform is not None)
        
        print(self._gpu_transform)
        print(crop_transform)
        print(img_transform)

        self._frame_reader = FrameReader(
            frame_dir, modality, crop_transform, img_transform, same_transform)
        
        self.list_index=[(1, 101), (101, 201), (201, 301), (301, 401), (401, 501), (501, 601), (601, 701), (701, 801), (801, 901), (901, 1001), (1001, 1101), (1101, 1201), (1201, 1301), (1301, 1401), (1401, 1501), (1501, 1601), (1601, 1701), (1701, 1801), (1801, 1901), (1901, 2001), (2001, 2101), (2101, 2201), (2201, 2301), (2301, 2401), (2401, 2501), (2501, 2601), (2601, 2701), (2701, 2801), (2801, 2901), (2901, 3001), (3001, 3101), (3101, 3201), (3201, 3301), (3301, 3401), (3401, 3501), (3501, 3601), (3601, 3701), (3701, 3801), (3801, 3901), (3901, 4001), (4001, 4101), (4101, 4201), (4201, 4301), (4301, 4401), (4401, 4501), (4501, 4601), (4601, 4701), (4701, 4801), (4801, 4901), (4901, 5001), (5001, 5101), (5101, 5201), (5201, 5301), (5301, 5401), (5401, 5501), (5501, 5601)]
        self.list_index_mixup = [(101, 201), (201, 301), (301, 401), (401, 501), (501, 601), (601, 701), (701, 801), (801, 901), (901, 1001), (1001, 1101), (1101, 1201), (1201, 1301), (1301, 1401), (1401, 1501), (1501, 1601), (1601, 1701), (1701, 1801), (1801, 1901), (1901, 2001), (2001, 2101), (2101, 2201), (2201, 2301), (2301, 2401), (2401, 2501), (2501, 2601), (2601, 2701), (2701, 2801), (2801, 2901), (2901, 3001), (3001, 3101), (3101, 3201), (3201, 3301), (3301, 3401), (3401, 3501), (3501, 3601), (3601, 3701), (3701, 3801), (3801, 3901), (3901, 4001), (4001, 4101), (4101, 4201), (4201, 4301), (4301, 4401), (4401, 4501), (4501, 4601), (4601, 4701), (4701, 4801), (4801, 4901), (4901, 5001), (5001, 5101), (5101, 5201), (5201, 5301), (5301, 5401), (5401, 5501), (5501, 5601), (1, 101)]
    def load_frame_gpu(self, batch, device):
        if self._gpu_transform is None:
            frame = batch['frame'].to(device)
        else:
            frame = _load_frame_deferred(self._gpu_transform, batch, device)
        return frame

    def _sample_uniform(self):
        video_meta = random.choices(
            self._labels, weights=self._weights_by_length)[0]

        video_len = video_meta['num_frames_2fps']
        base_idx = -self._pad_len * self._stride + random.randint(
            0, max(0, video_len - 1
                       + (2 * self._pad_len - self._clip_len) * self._stride))
        return video_meta, base_idx

    def _sample_foreground(self):
        video_idx, frame_idx = random.choices(self._flat_labels)[0]
        video_meta = self._labels[video_idx]
        video_len = video_meta['num_frames_2fps']

        lower_bound = max(
            -self._pad_len * self._stride,
            frame_idx - self._clip_len * self._stride + 1)
        upper_bound = min(
            video_len - 1 + (self._pad_len - self._clip_len) * self._stride,
            frame_idx)

        base_idx = random.randint(lower_bound, upper_bound) \
            if upper_bound > lower_bound else lower_bound

        assert base_idx <= frame_idx
        assert base_idx + self._clip_len > frame_idx
        return video_meta, base_idx

    def _get_one(self,index,mixup):
        if self._fg_upsample > 0 and random.random() >= self._fg_upsample:
            video_meta, base_idx = self._sample_foreground()
        else:
            video_meta, base_idx = self._sample_uniform()

        video_meta=self._labels[0]
        if mixup:
            base_idx=self.list_index_mixup[index][0]
        else:
            base_idx=self.list_index[index][0]
        labels = np.zeros(self._clip_len, np.int64)
        for event in video_meta['events']:
            event_frame = event['frame']

            # Index of event in label array
            label_idx = (event_frame - base_idx) // self._stride
            if (label_idx >= -self._dilate_len
                and label_idx < self._clip_len + self._dilate_len
            ):
                label = self._class_dict[event['label']]
                for i in range(
                    max(0, label_idx - self._dilate_len),
                    min(self._clip_len, label_idx + self._dilate_len + 1)
                ):
                    labels[i] = label

        
        start_time = time.time()
        # print(video_meta['video'],start_time)
        frames = self._frame_reader.load_frames(
            video_meta['video'], base_idx,
            base_idx + self._clip_len * self._stride, pad=True,
            stride=self._stride, randomize=not self._is_eval)
        # print(video_meta['video'],time.time() - start_time)
        # print("getone",mixup,index,len(frames),frames[0].shape,video_meta['video'],base_idx)
        return {'frame': frames, 'contains_event': int(np.sum(labels) > 0),
                'label': labels,'indexes':(index,base_idx)}

    def __getitem__(self, unused):
        ret = self._get_one(unused,False)

        if self._mixup:
            mix = self._get_one(unused,True)    # Sample another clip

            l = random.betavariate(0.2, 0.2)

            l=0.9
            label_dist = np.zeros((self._clip_len, len(self._class_dict) + 1))

            label_dist[range(self._clip_len), ret['label']] = l
            label_dist[range(self._clip_len), mix['label']] += 1. - l
            non_zero_indexes = np.nonzero(label_dist)

            if self._gpu_transform is None:
                ret['frame'] = l * ret['frame'] + (1. - l) * mix['frame']
            else:
                ret['mix_frame'] = mix['frame']
                ret['mix_weight'] = l

            ret['contains_event'] = max(
                ret['contains_event'], mix['contains_event'])
            ret['label'] = label_dist

        return ret

    def __len__(self):
        return self._dataset_len

    def print_info(self):
        _print_info_helper(self._src_file, self._labels)


class ActionSpotVideoDataset(Dataset):

    def __init__(
            self,
            classes,
            label_file,
            frame_dir,
            modality,
            clip_len,
            overlap_len=0,
            crop_dim=None,
            stride=1,
            pad_len=DEFAULT_PAD_LEN,
            flip=False,
            multi_crop=False,
            skip_partial_end=True
    ):
        self._src_file = label_file
        self._labels = load_json(label_file)
        self._class_dict = classes
        self._video_idxs = {x['video']: i for i, x in enumerate(self._labels)}
        self._clip_len = clip_len
        self._stride = stride

        crop_transform, img_transform = _get_img_transforms(
            is_eval=True, crop_dim=crop_dim, modality=modality, same_transform=True, multi_crop=multi_crop)

        # No need to enforce same_transform since the transforms are
        # deterministic
        self._frame_reader = FrameReader(
            frame_dir, modality, crop_transform, img_transform, False)

        self._flip = flip
        self._multi_crop = multi_crop

        self._clips = []
        for l in self._labels:
            has_clip = False
            for i in range(
                -pad_len * self._stride,
                max(0, l['num_frames_2fps'] - (overlap_len * stride)
                        * int(skip_partial_end)), \
                # Need to ensure that all clips have at least one frame
                (clip_len - overlap_len) * self._stride
            ):
                has_clip = True
                self._clips.append((l['video'], i))
            assert has_clip, l

    def __len__(self):
        return len(self._clips)

    def __getitem__(self, idx):
        video_name, start = self._clips[idx]
        frames = self._frame_reader.load_frames(
            video_name, start, start + self._clip_len * self._stride, pad=True,
            stride=self._stride)

        if self._flip:
            frames = torch.stack((frames, frames.flip(-1)), dim=0)

        return {'video': video_name, 'start': start // self._stride,
                'frame': frames}

    def get_labels(self, video):
        meta = self._labels[self._video_idxs[video]]
        num_frames = meta['num_frames_2fps']
        num_labels = num_frames // self._stride
        if num_frames % self._stride != 0:
            num_labels += 1
        labels = np.zeros(num_labels, np.int64)
        for event in meta['events']:
            frame = event['frame']
            if frame < num_frames:
                labels[frame // self._stride] = self._class_dict[event['label']]
            else:
                print('Warning: {} >= {} is past the end {}'.format(
                    frame, num_frames, meta['video']))
        return labels

    @property
    def augment(self):
        return self._flip or self._multi_crop

    @property
    def videos(self):
        # return [
        #     (v['video'], v['num_frames_2fps'] // self._stride,
        #      v['fps'] / self._stride) for v in self._labels[:1]]
        return sorted([
            (v['video'], v['num_frames_2fps'] // self._stride,
             v['fps'] / self._stride) for v in self._labels])

    @property
    def labels(self):
        assert self._stride > 0
        if self._stride == 1:
            return self._labels
        else:
            labels = []
            for x in self._labels:
                x_copy = copy.deepcopy(x)
                x_copy['fps'] /= self._stride
                x_copy['num_frames_2fps'] //= self._stride
                for e in x_copy['events']:
                    e['frame'] //= self._stride
                labels.append(x_copy)
            return labels

    def print_info(self):
        num_frames = sum([x['num_frames_2fps'] for x in self._labels])
        num_events = sum([len(x['events']) for x in self._labels])
        print('{} : {} videos, {} frames ({} stride), {:0.5f}% non-bg'.format(
            self._src_file, len(self._labels), num_frames, self._stride,
            num_events / num_frames * 100))


from nvidia.dali import pipeline_def,backend
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import nvidia.dali.plugin.pytorch as dalitorch
import tempfile
import cupy
from nvidia.dali.plugin.pytorch import LastBatchPolicy
import math
        
def distribute_elements(x,nb):
    quotient, remainder = divmod(x, nb)
    distribution = [quotient] * nb
    if remainder >0:
        for i in range(len(distribution)):
            distribution[i] += 1
    
    return distribution

class DaliDataSet(DALIGenericIterator):
    def __init__(
            self,
            epochs,
            batch_size, 
            output_map,
            devices,
            classes,                    # dict of class names to idx
            label_file,                 # path to label json
            modality,                   # [rgb, bw, flow]
            clip_len,
            dataset_len,                # Number of clips
            is_eval=True,               # Disable random augmentation
            crop_dim=None,
            stride=1,                   # Downsample frame rate
            same_transform=True,        # Apply the same random augmentation to
                                        # each frame in a clip
            dilate_len=0,               # Dilate ground truth labels
            mixup=False,
            pad_len=DEFAULT_PAD_LEN,    # Number of frames to pad the start
                                        # and end of videos
            fg_upsample=-1              # Sample foreground explicitly
        ):
        self._src_file = label_file
        self._labels = load_json(label_file)
        self._class_dict = classes

        #nouveau
        self.original_batch_size = batch_size

        if mixup:
            self.batch_size = 2 * batch_size
        else: 
            self.batch_size = batch_size

        self.batch_size_per_pipe = distribute_elements(self.batch_size,len(devices))
        #fin

        self.batch_size = batch_size
        self.nb_videos = dataset_len*2 if mixup else dataset_len
        self.mixup = mixup
        self.output_map = output_map
        self.devices=devices
        self.is_eval = is_eval
        
        epochs=50
        import math

        if is_eval:
            nb_clips_per_video = math.ceil(dataset_len/len(self._labels))*epochs
        else:
            nb_clips_per_video = math.ceil(dataset_len/len(self._labels))*epochs
        
        if mixup : nb_clips_per_video=nb_clips_per_video*2
        
        file_list_txt = ""
        for index,video in enumerate(self._labels):
            video_path = os.path.join("/home/ybenzakour/datasets/SoccerNet", video['video'] + ".mkv")
            for i in range(nb_clips_per_video):
                random_start = random.randint(1, video['num_frames_dali']-101)
                random_end= random_start+100
                file_list_txt += f"{video_path} {index} {random_start * 12} {(random_start+100) * 12}\n"
        
        tf = tempfile.NamedTemporaryFile()
        tf.write(str.encode(file_list_txt))
        tf.flush()

        self.pipes = [
            self.video_pipe(
                batch_size=self.batch_size_per_pipe[index], sequence_length = 100, stride_dali = 12, 
                step = -1, num_threads=8, device_id=i, file_list=tf.name,shard_id = index, 
                num_shards = len(devices)) for index,i in enumerate(devices)]
        # self.pipes = [
        #     self.video_pipe(
        #         batch_size=self.batch_size, sequence_length = 100, stride_dali = 12, 
        #                 step = -1, num_threads=8, device_id=i, file_list=tf.name) for i in devices
        #                 ]

        for pipe in self.pipes:
            pipe.build()
        super().__init__(self.pipes,output_map,size=self.nb_videos)
        # ,last_batch_policy=LastBatchPolicy.PARTIAL)
        self.gpu_transform = None
        self.device = torch.device('cuda:{}'.format(self.devices[1]))
        if not self.is_eval:
            self.gpu_transform = self.get_deferred_rgb_transform()

    def __next__(self):
        out = super().__next__()
        ret = self.getitem(out)
        if self.is_eval:
            frame = ret["frame"]
        else:
            frame = self.load_frame_deferred(self.gpu_transform, ret)
        return {"frame":frame,"label":ret["label"]}
    
    def delete(self):
        for pipe in self.pipes:
            pipe.__del__()
            del pipe
        backend.ReleaseUnusedMemory()

    
    def get_attr(self,batch):
        batch_labels = batch["label"]
        batch_images = batch["data"]
        sum_labels = torch.sum(batch_labels, dim=1 if len(batch_labels.shape)==2 else 0)
        contains_event = (sum_labels > 0).int()
        return {'frame': batch_images, 'contains_event': contains_event,
                    'label': batch_labels}
        # , 'start_frame_num' : batch['start_frame_num']}
    
    def move_to_device(self,batch):
        for key, tensor in batch.items():
                batch[key] = tensor.to(self.device)

    def getitem(self,data):
        nb_devices = len(self.devices)
        if(nb_devices >= 2):
            ret=self.get_attr(data[0])
            mix = self.get_attr(data[1])
            self.move_to_device(ret)
            self.move_to_device(mix)
      
        if(nb_devices >= 4):
            ret2 = self.get_attr(data[2])
            mix2 = self.get_attr(data[3])
            self.move_to_device(ret2)
            self.move_to_device(mix2)
        # ret=self.get_attr(data[0])
        # for key, tensor in ret.items():
        #         ret[key] = tensor.to(torch.device('cuda:{}'.format(self.devices[1])))
                # ret[key] = tensor.to(torch.device('cuda:0'))
        if self.mixup:
            if nb_devices == 4:
                for key, tensor in ret.items():
                    ret[key] = torch.cat((tensor,ret2[key]))
                for key, tensor in mix.items():
                    mix[key] = torch.cat((tensor,mix2[key]))
            # mix=self.get_attr(data[1])
            # for key, tensor in mix.items():
            #     mix[key] = tensor.to(torch.device('cuda:{}'.format(self.devices[1])))

            l = [random.betavariate(0.2, 0.2) for i in range(ret['frame'].shape[0])]

            # l=[0.9 for i in range(self.batch_size)]
            l = torch.tensor(l)
        #     print(l,l.device)
            label_dist = torch.zeros((ret['frame'].shape[0],100,len(self._class_dict) + 1),device=self.device)
            # label_dist = torch.zeros((self.batch_size,100,len(self._class_dict) + 1),device=torch.device('cuda:0'))
            for i in range(ret['frame'].shape[0]):
                label_dist[i,range(100), ret['label'][i]] = l[i].item()
                label_dist[i,range(100), mix['label'][i]] += 1. - l[i].item()

            if self.gpu_transform is None:
                for i in range(ret['frame'].shape[0]):
                    ret['frame'][i] = l[i].item() * ret['frame'][i] + (1. - l[i].item()) * mix['frame'][i]
            else:
                ret['mix_frame'] = mix['frame']
                ret['mix_weight'] = l

            ret['contains_event'] = torch.max(
                ret['contains_event'], mix['contains_event'])
            ret['label'] = label_dist
        return ret
    
    def get_deferred_rgb_transform(self):
        img_transforms = [
            # Jittering separately is faster (low variance)
            transforms.RandomApply(
                nn.ModuleList([transforms.ColorJitter(hue=0.2)]), p=0.25),
            transforms.RandomApply(
                nn.ModuleList([
                    transforms.ColorJitter(saturation=(0.7, 1.2))
                ]), p=0.25),
            transforms.RandomApply(
                nn.ModuleList([
                    transforms.ColorJitter(brightness=(0.7, 1.2))
                ]), p=0.25),
            transforms.RandomApply(
                nn.ModuleList([
                    transforms.ColorJitter(contrast=(0.7, 1.2))
                ]), p=0.25),

            # Jittering together is slower (high variance)
            # transforms.RandomApply(
            #     nn.ModuleList([
            #         transforms.ColorJitter(
            #             brightness=(0.7, 1.2), contrast=(0.7, 1.2),
            #             saturation=(0.7, 1.2), hue=0.2)
            #     ]), 0.8),

            transforms.RandomApply(
                nn.ModuleList([transforms.GaussianBlur(5)]), p=0.25),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ]
        return torch.jit.script(nn.Sequential(*img_transforms))
    
    def load_frame_deferred(self,gpu_transform, batch):
        frame = batch['frame']
        with torch.no_grad():
            for i in range(frame.shape[0]):
                frame[i] = gpu_transform(frame[i])

            if 'mix_weight' in batch:
                weight = batch['mix_weight'].to(self.device)
                # weight = batch['mix_weight'].to(torch.device('cuda:0'))
                frame *= weight[:, None, None, None, None]

                frame_mix = batch['mix_frame']
                for i in range(frame.shape[0]):
                    frame[i] += (1. - weight[i]) * gpu_transform(
                        frame_mix[i])
        
        return frame
    
    @pipeline_def
    def video_pipe(self,file_list,sequence_length,stride_dali,step,shard_id,num_shards):
        video, label, frame_num = fn.readers.video(
            device="gpu",
            file_list=file_list,
            sequence_length=sequence_length,
            #code de test
            random_shuffle=True,
            # random_shuffle=True,
            shard_id=shard_id,
            num_shards=num_shards,
            image_type=types.RGB,
            file_list_include_preceding_frame=True,
            file_list_frame_num=True,
            enable_frame_num = True,
            stride = stride_dali,
            step = step,
            pad_sequences=True,
            skip_vfr_check=True)
        if self.is_eval:
            video = fn.crop_mirror_normalize(video,
                                          dtype=types.FLOAT,
    #                                       output_layout="FHWC",
                                         output_layout = "FCHW",
                                        #  std=[255,255,255]
                                         mean = [IMAGENET_MEAN[i] * 255. for i in range(len(IMAGENET_MEAN))],
                                          std=[IMAGENET_STD[i] * 255. for i in range(len(IMAGENET_STD))]
                                         )
        else:
            video = fn.crop_mirror_normalize(video,
                                              dtype=types.FLOAT,
        #                                       output_layout="FHWC",
                                             output_layout = "FCHW",
                                             std=[255, 255, 255],
                                             mirror=fn.random.coin_flip(),
                                             )
        label = fn.python_function(label,frame_num,function=self.edit_labels,device="gpu")
        return video,label

    def edit_labels(self,label,frame_num):
        dilate_len=0
        clip_len=100
        video_meta = self._labels[label.item()]
    #     print(video_meta["video"])
        base_idx = frame_num.item()//12
        
        labels = cupy.zeros(100, np.int64)
        for event in video_meta['events']:
            event_frame = event['frame']
            # Index of event in label array
            label_idx = (event_frame - base_idx) // 1
            if (label_idx >= dilate_len
                and label_idx < clip_len + dilate_len
               ):
                label = self._class_dict[event['label']]
                for i in range(
                    max(0, label_idx - dilate_len),
                    min(clip_len, label_idx + dilate_len + 1)
                    ):
                    labels[i] = label
        return labels
    
    def print_info(self):
        _print_info_helper(self._src_file, self._labels)

class DaliDataSetVideo(DALIGenericIterator):

    def __init__(
            self,
            batch_size, 
            output_map,
            devices,
            classes,
            label_file,
            modality,
            clip_len,
            overlap_len=0,
            crop_dim=None,
            stride=1,
            pad_len=DEFAULT_PAD_LEN,
            flip=False,
            multi_crop=False,
            skip_partial_end=True
    ):
        self._src_file = label_file
        self._labels = load_json(label_file)
        self._class_dict = classes
        self._video_idxs = {x['video']: i for i, x in enumerate(self._labels)}
        self._clip_len = clip_len
        self._stride = stride

        self._flip = flip
        self._multi_crop = multi_crop

        self.batch_size = batch_size // len(devices)

        self._clips = []
        file_list_txt = ""
        cmp=0
        for l in self._labels:
            has_clip = False
            for i in range(
                1,l["num_frames_dali"], \
                # Need to ensure that all clips have at least one frame
                (clip_len - overlap_len) * self._stride
            ):
                if i+100>l['num_frames_dali']:
                    end = l['num_frames_base']
                else:
                    end = (i+100)*12
                has_clip = True
                self._clips.append((l['video'], i))
                video_path = os.path.join("/home/ybenzakour/datasets/SoccerNet", l['video'] + ".mkv")
                file_list_txt += f"{video_path} {cmp} {i * 12} {end}\n"
                cmp+=1
            assert has_clip, l
        
        tf = tempfile.NamedTemporaryFile()
        tf.write(str.encode(file_list_txt))
        tf.flush()

        self.pipes = [
            self.video_pipe(batch_size=self.batch_size, sequence_length = 100, stride_dali = 12, 
                        step = -1, num_threads=8, device_id=i, file_list=tf.name, shard_id = index, num_shards=len(devices)) for index,i in enumerate(devices)
                ]
        
        for pipe in self.pipes:
            pipe.build()
        size=len(self._clips)
        # size=57
        super().__init__(self.pipes,output_map,size=size)

        self.gpu_transform = None
        

    def __next__(self):
        out = super().__next__()
        video_names=[]
        starts=cupy.zeros(2*self.batch_size,np.int64)
        for i in range(out[0]["label"].shape[0]):
            video_name, start = self._clips[out[0]["label"][i]]
            video_names.append(video_name)
            starts[i]=start
        for i in range(out[1]["label"].shape[0]):
            video_name, start = self._clips[out[1]["label"][i]]
            video_names.append(video_name)
            starts[i+out[0]["label"].shape[0]]=start
        return {'video': video_names, 'start': torch.as_tensor(starts) ,
            'frame': torch.cat((out[0]["data"], out[1]["data"].to(torch.device('cuda'))), dim=0)}

    def delete(self):
        for pipe in self.pipes:
            pipe.__del__()
            del pipe
        backend.ReleaseUnusedMemory()
        
    @pipeline_def
    def video_pipe(self,file_list,sequence_length,stride_dali,step,shard_id,num_shards):
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
            stride = stride_dali,
            step = step,
            pad_sequences=True,
            skip_vfr_check=True)
        
        video = fn.crop_mirror_normalize(video,
                                          dtype=types.FLOAT,
    #                                       output_layout="FHWC",
                                         output_layout = "FCHW",
                                         mean = [IMAGENET_MEAN[i] * 255. for i in range(len(IMAGENET_MEAN))],
                                          std=[IMAGENET_STD[i] * 255. for i in range(len(IMAGENET_STD))]
                                         )
        return video,label
    
    def get_labels(self, video):
        meta = self._labels[self._video_idxs[video]]
        num_frames = meta['num_frames_dali']
        num_labels = num_frames // self._stride
        if num_frames % self._stride != 0:
            num_labels += 1
        labels = np.zeros(num_labels, np.int64)
        for event in meta['events']:
            frame = event['frame']
            if frame < num_frames:
                labels[frame // self._stride] = self._class_dict[event['label']]
            else:
                print('Warning: {} >= {} is past the end {}'.format(
                    frame, num_frames, meta['video']))
        return labels

    @property
    def augment(self):
        return self._flip or self._multi_crop

    @property
    def videos(self):
        # return [
        #     (v['video'], v['num_frames_dali'] // self._stride,
        #      v['fps'] / self._stride) for v in self._labels]
        return sorted([
            (v['video'], v['num_frames_dali'] // self._stride,
             v['fps'] / self._stride) for v in self._labels])

    @property
    def labels(self):
        assert self._stride > 0
        if self._stride == 1:
            return self._labels
        else:
            labels = []
            for x in self._labels:
                x_copy = copy.deepcopy(x)
                x_copy['fps'] /= self._stride
                x_copy['num_frames_dali'] //= self._stride
                for e in x_copy['events']:
                    e['frame'] //= self._stride
                labels.append(x_copy)
            return labels

    def print_info(self):
        num_frames = sum([x['num_frames_dali'] for x in self._labels])
        num_events = sum([len(x['events']) for x in self._labels])
        print('{} : {} videos, {} frames ({} stride), {:0.5f}% non-bg'.format(
            self._src_file, len(self._labels), num_frames, self._stride,
            num_events / num_frames * 100))