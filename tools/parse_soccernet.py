#!/usr/bin/env python3

import os
import argparse
import json
from collections import defaultdict

from SoccerNet.utils import getListGames

import math
import json
import cv2

def load_json(fpath):
    with open(fpath) as fp:
        return json.load(fp)
    
def get_stride(src_fps, sample_fps):
        sample_fps = sample_fps
        if sample_fps <= 0:
            stride = 1
        else:
            stride = int(src_fps / sample_fps)
        return stride
    
def read_fps(fps,sample_fps):
    stride = get_stride(fps,sample_fps)
    est_out_fps = fps / stride
    return est_out_fps

def get_num_frames(num_frames,fps, sample_fps):
    return math.ceil(num_frames/get_stride(fps,sample_fps))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('label_dir', type=str,
                        help='Path to the SoccerNetV2 labels')
    parser.add_argument('video_dir', type=str,
                        help='Path to extracted video frames')
    parser.add_argument('-o', '--out_dir', type=str,
                        help='Path to output parsed dataset')
    parser.add_argument('-i_fps', '--input_fps', type=int, default=25,
                        help='Sample input fps')
    parser.add_argument('-fps', '--wanted_sample_fps', type=int, default=2,
                        help='Sample output fps')
    parser.add_argument('-dali', '--dali', action='store_true')
    return parser.parse_args()


def load_split(split):
    if split == 'val':
        split = 'valid'

    videos = []
    for entry in getListGames(split):
        league, season, game = entry.split('/')
        videos.append((league, season, game))
    return videos


def get_label_names(labels):
    return {e['label'] for v in labels for e in v['events']}


def main(label_dir, video_dir, out_dir, input_fps, wanted_sample_fps, dali):
    labels_by_split = defaultdict(list)
    fpss=[]
    for split in ['train', 'val', 'test', 'challenge']:
        videos = load_split(split)
        for video in videos:
            league, season, game = video

            game = game.replace(' ','_')
            video_label_path = os.path.join(
                label_dir, league, season, game, 'Labels-v2.json')

            if split != 'challenge':
                video_labels = load_json(video_label_path)
            else:
                video_labels = {'annotations': []}

            num_events = 0
            for half in (1, 2):
                video_frame_dir = os.path.join(
                    video_dir, league, season, game, str(half)+"_224p")

                video_name = video_frame_dir + ".mkv"
                vc = cv2.VideoCapture(video_name)
                fps = vc.get(cv2.CAP_PROP_FPS)
                num_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
                # if(get_stride(fps,wanted_sample_fps)!=12):
                #     print(video_name)

                sample_fps = read_fps(fps,wanted_sample_fps if wanted_sample_fps < fps else fps)
                num_frames_after = get_num_frames(num_frames,fps,wanted_sample_fps if wanted_sample_fps < fps else fps)
                
                if dali :
                    if(get_stride(fps, wanted_sample_fps if wanted_sample_fps < fps else fps) != get_stride(input_fps,wanted_sample_fps)):
                        sample_fps=fps/get_stride(input_fps,wanted_sample_fps)
                        num_frames_dali = math.ceil(num_frames/get_stride(input_fps,wanted_sample_fps))
                    else:
                        num_frames_dali = num_frames_after

                video_id = '{}/{}/{}/{}'.format(league, season, game, str(half)+"_224p")
                
                half_events = []
                for label in video_labels['annotations']:
                    lhalf = int(label['gameTime'].split(' - ')[0])
                    if half == lhalf:
                        if dali:
                            if(get_stride(fps, wanted_sample_fps if wanted_sample_fps < fps else fps) != get_stride(input_fps,wanted_sample_fps)):
                                adj_frame = float(label['position']) / 1000 * (fps/get_stride(input_fps,wanted_sample_fps))
                            else:
                                adj_frame = float(label['position']) / 1000 * sample_fps
                        else:
                            adj_frame = float(label['position']) / 1000 * sample_fps
                        half_events.append({
                            'frame': int(adj_frame),
                            'label': label['label'],
                            'comment': '{}; {}'.format(
                                label['team'], label['visibility'])
                        })

                        if adj_frame >= num_frames_after:
                            print('Label past end: {} -- {} < {} -- {}'.format(
                                video_id, num_frames_after, int(adj_frame),
                                label['label']))
                num_events += len(half_events)
                half_events.sort(key=lambda x: x['frame'])

                # max_label_frame = max(e['frame'] for e in half_events) \
                #     if len(half_events) > 0 else 0
                # if max_label_frame >= num_frames:
                #     num_frames = max_label_frame + 1
                if dali:
                    labels_by_split[split].append({
                        'video': video_id,
                        'num_frames_2fps': num_frames_after,
                        'num_frames_dali': num_frames_dali,
                        'num_frames_base': num_frames,
                        'num_events': len(half_events),
                        'events': half_events,
                        'fps': sample_fps,
                        'width': 398,
                        'height': 224
                    })
                else : 
                    labels_by_split[split].append({
                        'video': video_id,
                        'num_frames': num_frames_after,
                        'num_frames_base': num_frames,
                        'num_events': len(half_events),
                        'events': half_events,
                        'fps': sample_fps,
                        'width': 398,
                        'height': 224
                    })
            assert len(video_labels['annotations']) == num_events, \
                video_label_path

    train_classes = get_label_names(labels_by_split['train'])
    assert train_classes == get_label_names(labels_by_split['test'])
    assert train_classes == get_label_names(labels_by_split['val'])

    print('Classes:', sorted(train_classes))

    for split, labels in labels_by_split.items():
        print('{} : {} videos : {} events'.format(
            split, len(labels), sum(len(l['events']) for l in labels)))
        labels.sort(key=lambda x: x['video'])

    if out_dir is not None:
        out_dir = os.path.join(out_dir,"{}_fps".format(wanted_sample_fps),"dali" if dali else "opencv")
        os.makedirs(out_dir, exist_ok=True)
        class_path = os.path.join(out_dir, 'class.txt')
        with open(class_path, 'w') as fp:
            fp.write('\n'.join(sorted(train_classes)))

        for split, labels in labels_by_split.items():
            out_path = os.path.join(out_dir, '{}.json'.format(split))
            with open(out_path, 'w') as fp:
                json.dump(labels, fp, indent=2, sort_keys=True)

    print('Done!')

    print(set(fpss))

if __name__ == '__main__':
    main(**vars(get_args()))
