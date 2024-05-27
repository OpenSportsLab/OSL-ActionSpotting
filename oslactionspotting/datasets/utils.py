from collections import defaultdict
import logging
import os
import cv2
import numpy as np
import torch
import math

from tqdm import tqdm

from oslactionspotting.core.utils.io import load_json


def get_stride(src_fps, sample_fps):
    """Get stride to apply based on the input and output fps.

    Args:
        src_fps (int): The input fps of the video.
        sample_fps (int): The output fps.
    Returns:
        stride (int): The stride to apply.
    """
    if sample_fps <= 0:
        stride = 1
    else:
        stride = int(src_fps / sample_fps)
    return stride


def read_fps(fps, sample_fps):
    """Computes the exact output fps based on input fps and wanted output fps.
    Example: if input fps is 25 and wanted output fps is 2, the exact output fps is 2.0833333333333335.

    Args:
        fps (int): The input fps.
        sample_fps (int): The wanted output fps.

    Returns:
        est_out_fps (float): The exact output fps.

    """
    stride = get_stride(fps, sample_fps)
    est_out_fps = fps / stride
    return est_out_fps


def get_num_frames(num_frames, fps, sample_fps):
    """Compute the number of frames of a video after fps changes.

    Args:
        num_frames (int): Number of frames in the base video.
        fps (int): The input fps.
        sample_fps (int): The output fps.

    Returns:
        (int): The number of frames with the output fps.
    """
    return math.ceil(num_frames / get_stride(fps, sample_fps))


def rulesToCombineShifts(shift_from_last_event, shift_until_next_event, params):
    """Set the rule to combine shifts based on two rules and parameters.

    Args:
        shift_from_last_event: First rule.
        shift_until_next_event: Second rule.
        params: Parameters to choose the rule.
    Returns:
        The rule.
    """
    s1 = shift_from_last_event
    s2 = shift_until_next_event
    K = params

    if s1 < K[2]:
        value = s1
    elif s1 < K[3]:
        if s2 <= K[0]:
            value = s1
        else:
            if (s1 - K[2]) / (K[3] - K[2]) < (K[1] - s2) / (K[1] - K[0]):
                value = s1
            else:
                value = s2
    else:
        value = s2

    return value


def oneHotToShifts(onehot, params):
    """

    Args:
        onehot: onehot vector of the shape (number of frames, number of actions).
        params: used to construct the shift.
    """
    nb_frames = onehot.shape[0]
    nb_actions = onehot.shape[1]

    Shifts = np.empty(onehot.shape)

    for i in range(nb_actions):

        x = onehot[:, i]
        K = params[:, i]
        shifts = np.empty(nb_frames)

        loc_events = np.where(x == 1)[0]
        nb_events = len(loc_events)

        if nb_events == 0:
            shifts = np.full(nb_frames, K[0])
        elif nb_events == 1:
            shifts = np.arange(nb_frames) - loc_events
        else:
            loc_events = np.concatenate(([-K[3]], loc_events, [nb_frames - K[0]]))
            for j in range(nb_frames):
                shift_from_last_event = j - loc_events[np.where(j >= loc_events)[0][-1]]
                shift_until_next_event = j - loc_events[np.where(j < loc_events)[0][0]]
                shifts[j] = rulesToCombineShifts(
                    shift_from_last_event, shift_until_next_event, K
                )

        Shifts[:, i] = shifts

    return Shifts


def getNegativeIndexes(labels, params, chunk_size):

    zero_one_labels = np.zeros(labels.shape)
    for i in np.arange(labels.shape[1]):
        zero_one_labels[:, i] = 1 - np.logical_or(
            np.where(labels[:, i] >= params[3, i], 1, 0),
            np.where(labels[:, i] <= params[0, i], 1, 0),
        )
    zero_one = np.where(np.sum(zero_one_labels, axis=1) > 0, 0, 1)

    zero_one_pad = np.append(
        np.append(
            [
                1 - zero_one[0],
            ],
            zero_one,
            axis=0,
        ),
        [1 - zero_one[-1]],
        axis=0,
    )
    zero_one_pad_shift = np.append(zero_one_pad[1:], zero_one_pad[-1])

    zero_one_sub = zero_one_pad - zero_one_pad_shift

    zero_to_one_index = np.where(zero_one_sub == -1)[0]
    one_to_zero_index = np.where(zero_one_sub == 1)[0]

    if zero_to_one_index[0] > one_to_zero_index[0]:
        one_to_zero_index = one_to_zero_index[1:]
    if zero_to_one_index.shape[0] > one_to_zero_index.shape[0]:
        zero_to_one_index = zero_to_one_index[:-1]

    list_indexes = list()

    for i, j in zip(zero_to_one_index, one_to_zero_index):
        if j - i >= chunk_size:
            list_indexes.append([i, j])

    return list_indexes


def getChunks_anchors(labels, game_index, params, chunk_size=240, receptive_field=80):

    # get indexes of labels
    indexes = list()
    for i in np.arange(labels.shape[1]):
        indexes.append(np.where(labels[:, i] == 0)[0].tolist())

    # Positive chunks
    anchors = list()

    class_counter = 0
    for event in indexes:
        for element in event:
            anchors.append([game_index, element, class_counter])
        class_counter += 1

    # Negative chunks

    negative_indexes = getNegativeIndexes(labels, params, chunk_size)

    for negative_index in negative_indexes:
        start = [negative_index[0], negative_index[1]]
        anchors.append([game_index, start, labels.shape[1]])

    return anchors


def getTimestampTargets(labels, num_detections):

    targets = np.zeros(
        (labels.shape[0], num_detections, 2 + labels.shape[-1]), dtype="float"
    )

    for i in np.arange(labels.shape[0]):

        time_indexes, class_values = np.where(labels[i] == 0)

        counter = 0

        for time_index, class_value in zip(time_indexes, class_values):

            # Confidence
            targets[i, counter, 0] = 1.0
            # frame index normalized
            targets[i, counter, 1] = time_index / (labels.shape[1])
            # The class one hot encoded
            targets[i, counter, 2 + class_value] = 1.0
            counter += 1

            if counter >= num_detections:
                print(
                    "More timestamp than what was fixed... A lot happened in that chunk"
                )
                break

    return targets


def timestamps2long(output_spotting, video_size, chunk_size, receptive_field):
    """Method to transform the timestamps to vectors"""
    start = 0
    last = False
    receptive_field = receptive_field // 2

    timestamps_long = (
        torch.zeros(
            [video_size, output_spotting.size()[-1] - 2],
            dtype=torch.float,
            device=output_spotting.device,
        )
        - 1
    )

    for batch in np.arange(output_spotting.size()[0]):

        tmp_timestamps = (
            torch.zeros(
                [chunk_size, output_spotting.size()[-1] - 2],
                dtype=torch.float,
                device=output_spotting.device,
            )
            - 1
        )

        for i in np.arange(output_spotting.size()[1]):
            tmp_timestamps[
                torch.floor(output_spotting[batch, i, 1] * (chunk_size - 1)).type(
                    torch.int
                ),
                torch.argmax(output_spotting[batch, i, 2:]).type(torch.int),
            ] = output_spotting[batch, i, 0]

        # ------------------------------------------
        # Store the result of the chunk in the video
        # ------------------------------------------

        # For the first chunk
        if start == 0:
            timestamps_long[0 : chunk_size - receptive_field] = tmp_timestamps[
                0 : chunk_size - receptive_field
            ]

        # For the last chunk
        elif last:
            timestamps_long[start + receptive_field : start + chunk_size] = (
                tmp_timestamps[receptive_field:]
            )
            break

        # For every other chunk
        else:
            timestamps_long[
                start + receptive_field : start + chunk_size - receptive_field
            ] = tmp_timestamps[receptive_field : chunk_size - receptive_field]

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


def batch2long(output_segmentation, video_size, chunk_size, receptive_field):
    """Method to transform the batches to vectors."""
    start = 0
    last = False
    receptive_field = receptive_field // 2

    segmentation_long = torch.zeros(
        [video_size, output_segmentation.size()[-1]],
        dtype=torch.float,
        device=output_segmentation.device,
    )

    for batch in np.arange(output_segmentation.size()[0]):

        tmp_segmentation = torch.nn.functional.one_hot(
            torch.argmax(output_segmentation[batch], dim=-1),
            num_classes=output_segmentation.size()[-1],
        )

        # ------------------------------------------
        # Store the result of the chunk in the video
        # ------------------------------------------

        # For the first chunk
        if start == 0:
            segmentation_long[0 : chunk_size - receptive_field] = tmp_segmentation[
                0 : chunk_size - receptive_field
            ]

        # For the last chunk
        elif last:
            segmentation_long[start + receptive_field : start + chunk_size] = (
                tmp_segmentation[receptive_field:]
            )
            break

        # For every other chunk
        else:
            segmentation_long[
                start + receptive_field : start + chunk_size - receptive_field
            ] = tmp_segmentation[receptive_field : chunk_size - receptive_field]

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




def feats2clip(
    feats, stride, clip_length, padding="replicate_last", off=0, modif_last_index=False
):
    """Converts a sequence of feature vectors into a sequence of overlapping clips.
    Args:
        feats: A tensor of shape (num_frames, feature_dim), representing the input feature vectors.
        stride: The step size between the starting points of consecutive clips.
        clip_length: The number of frames in each clip.
        padding: The padding strategy, either "zeropad" or "replicate_last".
        off: An offset to adjust the starting points of clips.
        modif_last_index: A flag indicating whether to modify the last index to ensure the last clip is aligned with the end of feats.
    """
    if padding == "zeropad":
        print("beforepadding", feats.shape)
        pad = feats.shape[0] - int(feats.shape[0] / stride) * stride
        print("pad need to be", clip_length - pad)
        m = torch.nn.ZeroPad2d((0, 0, clip_length - pad, 0))
        feats = m(feats)
        print("afterpadding", feats.shape)
        # nn.ZeroPad2d(2)
    if modif_last_index:
        off = 0
    idx = torch.arange(start=0, end=feats.shape[0] - 1, step=stride)
    idxs = []
    for i in torch.arange(-off, clip_length - off):
        idxs.append(idx + i)
    idx = torch.stack(idxs, dim=1)

    if padding == "replicate_last":
        idx = idx.clamp(0, feats.shape[0] - 1)
    if modif_last_index:
        idx[-1] = torch.arange(clip_length) + feats.shape[0] - clip_length
        return feats[idx, :]
    # print(idx)
    return feats[idx, ...]

def get_num_frames(num_frames, fps, sample_fps):
    """Compute the number of frames of a video after fps changes.

    Args:
        num_frames (int): Number of frames in the base video.
        fps (int): The input fps.
        sample_fps (int): The output fps.

    Returns:
        (int): The number of frames with the output fps.
    """
    return math.ceil(num_frames / get_stride(fps, sample_fps))

def annotationstoe2eformat(label_files, video_dirs, input_fps, extract_fps, dali):
    """Adapt annotations jsons to e2e format.

    Args:
        label_files (string,list[string]): Json files of annotations.
        label_dirs (string,list[string]): Data root folder of videos. Must match number of label files.
        input_fps (int): Fps of input videos.
        extract_fps (int): Fps at which we extract frames.
        dali (bool): WHether processing with dali or opencv.
    """
    if not isinstance(label_files,list):
        label_files = [label_files]
    if not isinstance(video_dirs,list):
        video_dirs = [video_dirs]
    assert len(label_files) == len(video_dirs)

    labels_e2e = list()
    classes_by_label_dir = []
    for label_dir, video_dir in zip(label_files, video_dirs):
        logging.info('Processing '+ label_dir + ' to e2e format.')
        videos = []
        annotations = load_json(label_dir)
        labels = annotations["labels"]
        classes_by_label_dir.append(labels)
        videos = annotations["videos"]
        for video in tqdm(videos):
            if "annotations" in video.keys():
                video_annotations = video["annotations"]
            else:
                video_annotations = []

            num_events = 0

            vc = cv2.VideoCapture(os.path.join(video_dir, video["path"]))
            width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = vc.get(cv2.CAP_PROP_FPS)
            num_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))

            sample_fps = read_fps(
                fps, extract_fps if extract_fps < fps else fps
            )
            num_frames_after = get_num_frames(
                num_frames, fps, extract_fps if extract_fps < fps else fps
            )

            if dali:
                if get_stride(
                    fps, extract_fps if extract_fps < fps else fps
                ) != get_stride(input_fps, extract_fps):
                    sample_fps = fps / get_stride(input_fps, extract_fps)
                    num_frames_dali = math.ceil(
                        num_frames / get_stride(input_fps, extract_fps)
                    )
                else:
                    num_frames_dali = num_frames_after

            # video_id = os.path.splitext(video["path"])[0]
            video_id = os.path.join(video_dir, video["path"])

            events = []
            for annotation in video_annotations:
                if dali:
                    if get_stride(
                        fps, extract_fps if extract_fps < fps else fps
                    ) != get_stride(input_fps, extract_fps):
                        adj_frame = (
                            float(annotation["position"])
                            / 1000
                            * (fps / get_stride(input_fps, extract_fps))
                        )
                    else:
                        adj_frame = float(annotation["position"]) / 1000 * sample_fps
                    if int(adj_frame) == 0:
                        adj_frame = 1
                else:
                    adj_frame = float(annotation["position"]) / 1000 * sample_fps
                events.append(
                    {
                        "frame": int(adj_frame),
                        "label": annotation["label"],
                        "team": annotation["team"],
                        "visibility": annotation["visibility"],
                    }
                )

            num_events += len(events)
            events.sort(key=lambda x: x["frame"])

            
            labels_e2e.append(
                {   
                    "events": events,
                    "fps": sample_fps,
                    "num_frames": num_frames_dali if dali else num_frames_after,
                    "num_frames_base": num_frames,
                    "num_events": len(events),
                    "width": width,
                    "height": height,
                    "video": video_id,
                    "path" : video["path"]
                }
            )
        assert len(video_annotations) == num_events
    classes = classes_by_label_dir[0]
    for classes_tmp in classes_by_label_dir:
        assert classes == classes_tmp
    labels_e2e.sort(key=lambda x: x["video"])
    return labels_e2e
