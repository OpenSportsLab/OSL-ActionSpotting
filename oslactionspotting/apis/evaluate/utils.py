"""
Copyright 2022 James Hong, Haotian Zhang, Matthew Fisher, Michael Gharbi,
Kayvon Fatahalian

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import time
import copy
from collections import defaultdict
import json
import logging
import os
import zipfile
import numpy as np
from tabulate import tabulate
from tqdm import tqdm
from oslactionspotting.core.utils.io import store_json


def non_maximum_supression(pred, window):
    """Non maximum suppression for predictions for a window size

    Args:
        pred (List[dict]): List of dictionnaries that contain predictions per video
        window (int): The window size between frames.

    Returns:
        new_pred (List[dict]) : The predictions after the nms.
    """
    new_pred = []
    for video_pred in pred:
        events_by_label = defaultdict(list)
        for e in video_pred["events"]:
            events_by_label[e["label"]].append(e)

        events = []
        for v in events_by_label.values():
            for e1 in v:
                for e2 in v:
                    if (
                        e1["frame"] != e2["frame"]
                        and abs(e1["frame"] - e2["frame"]) <= window
                        and e1["confidence"] < e2["confidence"]
                    ):
                        # Found another prediction in the window that has a
                        # higher score
                        break
                else:
                    events.append(e1)
        events.sort(key=lambda x: x["frame"])
        new_video_pred = copy.deepcopy(video_pred)
        new_video_pred["events"] = events
        new_video_pred["num_events"] = len(events)
        new_pred.append(new_video_pred)
    return new_pred


def store_eval_files_json(raw_pred, eval_dir):
    """Given raw predictions, store prediction for each video in different files.

    Args:
        raw_pred (list[dict]): List of predictions for each video.
        eval_dir (string) : Path base under which the pred files will be stored.

    Returns:
        only_one_file (bool): If raw predictions contains only one video.
    """
    only_one_file = False
    video_pred = defaultdict(list)
    video_fps = defaultdict(list)
    for obj in raw_pred:
        video = obj["video"]
        video_fps[video] = obj["fps"]
        for event in obj["events"]:
            video_pred[video].append(
                {
                    "frame": event["frame"],
                    "label": event["label"],
                    "confidence": event["confidence"],
                    "position": event["position"],
                    "gameTime": event["gameTime"],
                }
            )

    for video, pred in video_pred.items():
        # if video.startswith('/'):
        #     video = video[1:]
        if len(raw_pred) == 1:
            video_out_dir = eval_dir
            only_one_file = True
        else:
            video_out_dir = os.path.join(eval_dir, os.path.splitext(video)[0])
        os.makedirs(video_out_dir, exist_ok=True)
        store_json(
            os.path.join(video_out_dir, "results_spotting.json"),
            {"Url": video, "predictions": pred, "fps": video_fps[video]},
            pretty=True,
        )
    logging.info("Stored to {}".format(eval_dir))
    return only_one_file


def label2vector(
    labels, num_classes=17, framerate=2, EVENT_DICTIONARY={}, vector_size=None
):
    """Transform list of dict containing ground truth labels into a 2D array.
    Args:
        labels (List[dict]): List of groundtruth labels for a video.
        num_classes (int): Number of classes.
            Default: 17.
        framerate (int): Rate at which the frames have been processed in a video.
            Default: 2.
        EVENT_DICTIONARY (dict): Mapping between classes and indexes.
            Default: {}.
        vector_size (int): Size of the returned vector.
            Default: None.

    Returns:
        dense_labels (Numpy array): Array of shape (vector_size, num_classes) containing a value.
            The first index is the frame number and the second one is the class. The value can be -1 is the event
            is unshown, 1 is the event is visible and 0 otherwise, meaning that the event do not occur for this frame.

    """
    vector_size = int(90 * 60 * framerate if vector_size is None else vector_size)

    dense_labels = np.zeros((vector_size, num_classes))

    for annotation in labels:

        event = annotation["label"]
        if "frame" in annotation:
            frame = int(annotation["frame"])

        else:
            time = annotation["gameTime"]

            minutes = int(time[-5:-3])
            seconds = int(time[-2::])
            # annotation at millisecond precision
            if "position" in annotation:
                frame = int(framerate * (int(annotation["position"]) / 1000))
            # annotation at second precision
            else:
                frame = framerate * (seconds + 60 * minutes)

        label = EVENT_DICTIONARY[event]

        value = 1
        if "visibility" in annotation.keys():
            if annotation["visibility"] == "not shown":
                value = -1

        frame = min(frame, vector_size - 1)
        dense_labels[frame][label] = value

    return dense_labels


def predictions2vector(
    predictions, num_classes=17, framerate=2, EVENT_DICTIONARY={}, vector_size=None
):
    """Transform list of dict containing predictions into a 2D array.
    Args:
        predictions (List[dict]): List of predictions for a video.
        num_classes (int): Number of classes.
            Default: 17.
        framerate (int): Rate at which the frames have been processed in a video.
            Default: 2.
        EVENT_DICTIONARY (dict): Mapping between classes and indexes.
            Default: {}.
        vector_size (int): Size of the returned vector.
            Default: None.

    Returns:
        dense_predictions (Numpy array): Array of shape (vector_size, num_classes) containing a value.
            The first index is the frame number and the second one is the class. The value is the score of the class.

    """
    vector_size = int(90 * 60 * framerate if vector_size is None else vector_size)

    dense_predictions = np.zeros((vector_size, num_classes)) - 1

    for annotation in predictions:

        event = annotation["label"]

        if "frame" in annotation:
            frame = int(annotation["frame"])
        else:
            time = int(annotation["position"])

            # half = int(annotation["half"])

            frame = int(framerate * (time / 1000))

        label = EVENT_DICTIONARY[event]

        frame = min(frame, vector_size - 1)
        dense_predictions[frame][label] = annotation["confidence"]

    return dense_predictions


np.seterr(divide="ignore", invalid="ignore")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_class_scores(target, closest, detection, delta):
    """Compute the scores for a single class.

    Args:
        target (np.array(vector_size): List of ground truth targets of shape (number of frames).
        closest (np.array(vector_size - 1): List of closest action index of shape (number of frames - 1).
        detection (np.array(vector_size): List of predictions of shape (number of frames).
        delta (int): Tolerance.

    Returns:
        game_detections (np.array(number of predictions,3): Array to save the scores with the first column being
        the prediction score, the second one being 1 if an event have been found with requirements, the third one being
        the score of the closest action.
        len(gt_indexes_visible) (int): number of visible events.
        len(gt_indexes_unshown) (int): number of unshown events.

    """

    # Retrieving the important variables
    gt_indexes = np.where(target != 0)[0]
    gt_indexes_visible = np.where(target > 0)[0]
    gt_indexes_unshown = np.where(target < 0)[0]
    pred_indexes = np.where(detection >= 0)[0]
    pred_scores = detection[pred_indexes]

    # Array to save the results, each is [pred_scor,{1 or 0},closest[pred_indexes]]
    game_detections = np.zeros((len(pred_indexes), 3))
    game_detections[:, 0] = np.copy(pred_scores)
    game_detections[:, 2] = np.copy(closest[pred_indexes])

    remove_indexes = list()

    for gt_index in gt_indexes:

        max_score = -1
        max_index = None
        game_index = 0
        selected_game_index = 0

        for pred_index, pred_score in zip(pred_indexes, pred_scores):

            if pred_index < gt_index - delta:
                game_index += 1
                continue
            if pred_index > gt_index + delta:
                break

            if (
                abs(pred_index - gt_index) <= delta / 2
                and pred_score > max_score
                and pred_index not in remove_indexes
            ):
                max_score = pred_score
                max_index = pred_index
                selected_game_index = game_index
            game_index += 1

        if max_index is not None:
            game_detections[selected_game_index, 1] = 1
            remove_indexes.append(max_index)

    return game_detections, len(gt_indexes_visible), len(gt_indexes_unshown)


def compute_precision_recall_curve(targets, closests, detections, delta):
    """Compute precision recall curve.

    Args:
        target (List(np.array(vector_size,num_classes)): List of ground truth targets of shape (number of videos, number of frames,number of classes).
        closest (List(np.array(vector_size - 1,num_classes)): List of closest action index of shape (number of videos, number of frames - 1,number of classes).
        detection (List(np.array(vector_size,num_classes)): List of predictions of shape (number of videos, number of frames,number of classes).
        delta (int): Tolerance.

    Returns:
        precision: List of precision points.
        recall: List of recall points.
        precision_visible: List of precision points for only the visible events.
        recall_visible: List of recall points for only the visible events.
        precision_unshown: List of precision points for only the unshown events.
        recall_unshown: List of recall points for only the unshown events.

    """

    # Store the number of classes
    num_classes = targets[0].shape[-1]

    # 200 confidence thresholds between [0,1]
    thresholds = np.linspace(0, 1, 200)

    # Store the precision and recall points
    precision = list()
    recall = list()
    precision_visible = list()
    recall_visible = list()
    precision_unshown = list()
    recall_unshown = list()

    # Apply Non-Maxima Suppression if required
    start = time.time()

    # Precompute the predictions scores and their correspondence {TP, FP} for each class
    for c in np.arange(num_classes):
        total_detections = np.zeros((1, 3))
        total_detections[0, 0] = -1
        n_gt_labels_visible = 0
        n_gt_labels_unshown = 0

        # Get the confidence scores and their corresponding TP or FP characteristics for each game
        for target, closest, detection in zip(targets, closests, detections):
            tmp_detections, tmp_n_gt_labels_visible, tmp_n_gt_labels_unshown = (
                compute_class_scores(
                    target[:, c], closest[:, c], detection[:, c], delta
                )
            )
            total_detections = np.append(total_detections, tmp_detections, axis=0)
            n_gt_labels_visible = n_gt_labels_visible + tmp_n_gt_labels_visible
            n_gt_labels_unshown = n_gt_labels_unshown + tmp_n_gt_labels_unshown

        precision.append(list())
        recall.append(list())
        precision_visible.append(list())
        recall_visible.append(list())
        precision_unshown.append(list())
        recall_unshown.append(list())

        # Get only the visible or unshown actions
        total_detections_visible = np.copy(total_detections)
        total_detections_unshown = np.copy(total_detections)
        total_detections_visible[
            np.where(total_detections_visible[:, 2] <= 0.5)[0], 0
        ] = -1
        total_detections_unshown[
            np.where(total_detections_unshown[:, 2] >= -0.5)[0], 0
        ] = -1

        # Get the precision and recall for each confidence threshold
        for threshold in thresholds:
            pred_indexes = np.where(total_detections[:, 0] >= threshold)[0]
            pred_indexes_visible = np.where(
                total_detections_visible[:, 0] >= threshold
            )[0]
            pred_indexes_unshown = np.where(
                total_detections_unshown[:, 0] >= threshold
            )[0]
            TP = np.sum(total_detections[pred_indexes, 1])
            TP_visible = np.sum(total_detections[pred_indexes_visible, 1])
            TP_unshown = np.sum(total_detections[pred_indexes_unshown, 1])
            p = np.nan_to_num(TP / len(pred_indexes))
            r = np.nan_to_num(TP / (n_gt_labels_visible + n_gt_labels_unshown))
            p_visible = np.nan_to_num(TP_visible / len(pred_indexes_visible))
            r_visible = np.nan_to_num(TP_visible / n_gt_labels_visible)
            p_unshown = np.nan_to_num(TP_unshown / len(pred_indexes_unshown))
            r_unshown = np.nan_to_num(TP_unshown / n_gt_labels_unshown)
            precision[-1].append(p)
            recall[-1].append(r)
            precision_visible[-1].append(p_visible)
            recall_visible[-1].append(r_visible)
            precision_unshown[-1].append(p_unshown)
            recall_unshown[-1].append(r_unshown)

    precision = np.array(precision).transpose()
    recall = np.array(recall).transpose()
    precision_visible = np.array(precision_visible).transpose()
    recall_visible = np.array(recall_visible).transpose()
    precision_unshown = np.array(precision_unshown).transpose()
    recall_unshown = np.array(recall_unshown).transpose()

    # Sort the points based on the recall, class per class
    for i in np.arange(num_classes):
        index_sort = np.argsort(recall[:, i])
        precision[:, i] = precision[index_sort, i]
        recall[:, i] = recall[index_sort, i]

    # Sort the points based on the recall, class per class
    for i in np.arange(num_classes):
        index_sort = np.argsort(recall_visible[:, i])
        precision_visible[:, i] = precision_visible[index_sort, i]
        recall_visible[:, i] = recall_visible[index_sort, i]

    # Sort the points based on the recall, class per class
    for i in np.arange(num_classes):
        index_sort = np.argsort(recall_unshown[:, i])
        precision_unshown[:, i] = precision_unshown[index_sort, i]
        recall_unshown[:, i] = recall_unshown[index_sort, i]

    return (
        precision,
        recall,
        precision_visible,
        recall_visible,
        precision_unshown,
        recall_unshown,
    )


def compute_mAP(precision, recall):
    """Compute mean AP per class from precision and recall points

    Args:
        precision: List of precision points.
        recall: List of recall points.

    Returns:
        np.mean(mAP_per_class): mean of the mAP over all classes.
        mAP_per_class: List of mAP for each class.

    """
    # Array for storing the AP per class
    AP = np.array([0.0] * precision.shape[-1])

    # Loop for all classes
    for i in np.arange(precision.shape[-1]):

        # 11 point interpolation
        for j in np.arange(11) / 10:

            index_recall = np.where(recall[:, i] >= j)[0]

            possible_value_precision = precision[index_recall, i]
            max_value_precision = 0

            if possible_value_precision.shape[0] != 0:
                max_value_precision = np.max(possible_value_precision)

            AP[i] += max_value_precision

    mAP_per_class = AP / 11

    return np.mean(mAP_per_class), mAP_per_class


# Tight: (SNv3): np.arange(5)*1 + 1
# Loose: (SNv1/v2): np.arange(12)*5 + 5


def delta_curve(targets, closests, detections, framerate, deltas=np.arange(5) * 1 + 1):
    """Compute lists of mAP for each tolerance.

    Args:
        targets (List(np.array(vector_size,num_classes)): List of ground truth targets of shape (number of videos, number of frames,number of classes).
        closests (List(np.array(vector_size - 1,num_classes)): List of closest action index of shape (number of videos, number of frames - 1,number of classes).
        detections (List(np.array(vector_size,num_classes)): List of predictions of shape (number of videos, number of frames,number of classes).
        delta (np.array): Tolerances.

    Returns:
        mAP: List of mean mAP over all the classes for each tolerance.
        mAP_per_class: List of list of mAP for all classes for each tolerance.
        mAP_visible: List of mean mAP over all the classes for each tolerance only for the visible events.
        mAP_per_class_visible: List of list of mAP for all classes for each tolerance only for the visible events.
        mAP_unshown: List of mean mAP over all the classes for each tolerance only for the unshown events.
        mAP_per_class_unshown: List of list of mAP for all classes for each tolerance only for the unshown events.

    """
    mAP = list()
    mAP_per_class = list()
    mAP_visible = list()
    mAP_per_class_visible = list()
    mAP_unshown = list()
    mAP_per_class_unshown = list()

    for delta in tqdm(deltas * framerate):

        (
            precision,
            recall,
            precision_visible,
            recall_visible,
            precision_unshown,
            recall_unshown,
        ) = compute_precision_recall_curve(targets, closests, detections, delta)

        tmp_mAP, tmp_mAP_per_class = compute_mAP(precision, recall)
        mAP.append(tmp_mAP)
        mAP_per_class.append(tmp_mAP_per_class)
        # TODO: compute visible/undshown from another JSON file containing only the visible/unshown annotations
        tmp_mAP_visible, tmp_mAP_per_class_visible = compute_mAP(
            precision_visible, recall_visible
        )
        mAP_visible.append(tmp_mAP_visible)
        mAP_per_class_visible.append(tmp_mAP_per_class_visible)
        tmp_mAP_unshown, tmp_mAP_per_class_unshown = compute_mAP(
            precision_unshown, recall_unshown
        )
        mAP_unshown.append(tmp_mAP_unshown)
        mAP_per_class_unshown.append(tmp_mAP_per_class_unshown)

    return (
        mAP,
        mAP_per_class,
        mAP_visible,
        mAP_per_class_visible,
        mAP_unshown,
        mAP_per_class_unshown,
    )


def average_mAP(
    targets, detections, closests, framerate=2, deltas=np.arange(5) * 1 + 1
):
    """Compute average mAP.

    Args:
        targets (List(np.array(vector_size,num_classes)): List of ground truth targets of shape (number of videos, number of frames,number of classes).
        closests (List(np.array(vector_size - 1,num_classes)): List of closest action index of shape (number of videos, number of frames - 1,number of classes).
        detections (List(np.array(vector_size,num_classes)): List of predictions of shape (number of videos, number of frames,number of classes).
        framerate (int).
            Default: 2.
        delta (np.array): Tolerances.

    Returns:
        a_mAP: average mAP.
        a_mAP_per_class: List of average mAP for all classes.
        a_mAP_visible: average mAP only for the visible events.
        a_mAP_per_class_visible: List of average mAP only for the visible events.
        a_mAP_unshown: average mAP only for the unshown events.
        a_mAP_per_class_unshown: List of average mAP only for the unshown events.

    """
    (
        mAP,
        mAP_per_class,
        mAP_visible,
        mAP_per_class_visible,
        mAP_unshown,
        mAP_per_class_unshown,
    ) = delta_curve(targets, closests, detections, framerate, deltas)

    if len(mAP) == 1:
        return (
            mAP[0],
            mAP_per_class[0],
            mAP_visible[0],
            mAP_per_class_visible[0],
            mAP_unshown[0],
            mAP_per_class_unshown[0],
        )

    # Compute the average mAP
    integral = 0.0
    for i in np.arange(len(mAP) - 1):
        integral += (mAP[i] + mAP[i + 1]) / 2
    a_mAP = integral / ((len(mAP) - 1))

    integral_visible = 0.0
    for i in np.arange(len(mAP_visible) - 1):
        integral_visible += (mAP_visible[i] + mAP_visible[i + 1]) / 2
    a_mAP_visible = integral_visible / ((len(mAP_visible) - 1))

    integral_unshown = 0.0
    for i in np.arange(len(mAP_unshown) - 1):
        integral_unshown += (mAP_unshown[i] + mAP_unshown[i + 1]) / 2
    a_mAP_unshown = integral_unshown / ((len(mAP_unshown) - 1))
    a_mAP_unshown = a_mAP_unshown * 17 / 13

    a_mAP_per_class = list()
    for c in np.arange(len(mAP_per_class[0])):
        integral_per_class = 0.0
        for i in np.arange(len(mAP_per_class) - 1):
            integral_per_class += (mAP_per_class[i][c] + mAP_per_class[i + 1][c]) / 2
        a_mAP_per_class.append(integral_per_class / ((len(mAP_per_class) - 1)))

    a_mAP_per_class_visible = list()
    for c in np.arange(len(mAP_per_class_visible[0])):
        integral_per_class_visible = 0.0
        for i in np.arange(len(mAP_per_class_visible) - 1):
            integral_per_class_visible += (
                mAP_per_class_visible[i][c] + mAP_per_class_visible[i + 1][c]
            ) / 2
        a_mAP_per_class_visible.append(
            integral_per_class_visible / ((len(mAP_per_class_visible) - 1))
        )

    a_mAP_per_class_unshown = list()
    for c in np.arange(len(mAP_per_class_unshown[0])):
        integral_per_class_unshown = 0.0
        for i in np.arange(len(mAP_per_class_unshown) - 1):
            integral_per_class_unshown += (
                mAP_per_class_unshown[i][c] + mAP_per_class_unshown[i + 1][c]
            ) / 2
        a_mAP_per_class_unshown.append(
            integral_per_class_unshown / ((len(mAP_per_class_unshown) - 1))
        )

    return (
        a_mAP,
        a_mAP_per_class,
        a_mAP_visible,
        a_mAP_per_class_visible,
        a_mAP_unshown,
        a_mAP_per_class_unshown,
    )


def LoadJsonFromZip(zippedFile, JsonPath):
    with zipfile.ZipFile(zippedFile, "r") as z:
        # print(filename)
        with z.open(JsonPath) as f:
            data = f.read()
            d = json.loads(data.decode("utf-8"))

    return d


def get_closest_action_index(dense_labels, closest_numpy):
    """Assign the action to a list of indexes based on the previous and the next indexes.
    Example:
        If action occurs at index 10, the previous index at which the actions occured is 6 and the next one is 12,
        the action will then be assigned from index 8 to 11.

    Args:
        dense_labels: Vector of groundtruth labels.
        closest_numpy: Empty vector that stores the labels at the closest indexes.

    Returns:
        closest_numpy: Vector that stores the labels at the closest indexes.
    """
    for c in np.arange(dense_labels.shape[-1]):
        indexes = np.where(dense_labels[:, c] != 0)[0].tolist()
        if len(indexes) == 0:
            continue
        indexes.insert(0, -indexes[0])
        indexes.append(2 * closest_numpy.shape[0])
        for i in np.arange(len(indexes) - 2) + 1:
            start = max(0, (indexes[i - 1] + indexes[i]) // 2)
            stop = min(closest_numpy.shape[0], (indexes[i] + indexes[i + 1]) // 2)
            closest_numpy[start:stop, c] = dense_labels[indexes[i], c]
    return closest_numpy


def compute_performances_mAP(
    metric, targets_numpy, detections_numpy, closests_numpy, INVERSE_EVENT_DICTIONARY
):
    """Compute the different mAP and display them.

    Args:
        metric (string): The metric that will be used to compute the mAP.
        targets_numpy (List(np.array(vector_size,num_classes)): List of ground truth targets of shape (number of videos, number of frames,number of classes).
        detections_numpy (List(np.array(vector_size - 1,num_classes)): List of closest action index of shape (number of videos, number of frames - 1,number of classes).
        closests_numpy (List(np.array(vector_size,num_classes)): List of predictions of shape (number of videos, number of frames,number of classes).
        INVERSE_EVENT_DICTIONARY (dict): mapping between indexes and class name.

    Returns:
        results (dict): Dictionnary containing the different mAP computed.
    """
    if metric == "loose":
        deltas = np.arange(12) * 5 + 5
    elif metric == "tight":
        deltas = np.arange(5) * 1 + 1
    elif metric == "at1":
        deltas = np.array([1])
    elif metric == "at2":
        deltas = np.array([2])
    elif metric == "at3":
        deltas = np.array([3])
    elif metric == "at4":
        deltas = np.array([4])
    elif metric == "at5":
        deltas = np.array([5])

    # Compute the performances
    (
        a_mAP,
        a_mAP_per_class,
        a_mAP_visible,
        a_mAP_per_class_visible,
        a_mAP_unshown,
        a_mAP_per_class_unshown,
    ) = average_mAP(
        targets_numpy, detections_numpy, closests_numpy, framerate=2, deltas=deltas
    )

    results = {
        "a_mAP": a_mAP,
        "a_mAP_per_class": a_mAP_per_class,
        "a_mAP_visible": a_mAP_visible,
        "a_mAP_per_class_visible": a_mAP_per_class_visible,
        "a_mAP_unshown": a_mAP_unshown,
        "a_mAP_per_class_unshown": a_mAP_per_class_unshown,
    }

    rows = []
    for i in range(len(results["a_mAP_per_class"])):
        label = INVERSE_EVENT_DICTIONARY[i]
        rows.append(
            (
                label,
                "{:0.2f}".format(results["a_mAP_per_class"][i] * 100),
                "{:0.2f}".format(results["a_mAP_per_class_visible"][i] * 100),
                "{:0.2f}".format(results["a_mAP_per_class_unshown"][i] * 100),
            )
        )
    rows.append(
        (
            "Average mAP",
            "{:0.2f}".format(results["a_mAP"] * 100),
            "{:0.2f}".format(results["a_mAP_visible"] * 100),
            "{:0.2f}".format(results["a_mAP_unshown"] * 100),
        )
    )

    logging.info("Best Performance at end of training ")
    logging.info("Metric: " + metric)
    logging.info("\n" + tabulate(rows, headers=["", "Any", "Visible", "Unseen"]))
    # print(tabulate(rows, headers=['', 'Any', 'Visible', 'Unseen']))
    return results
