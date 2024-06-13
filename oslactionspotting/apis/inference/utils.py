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

from collections import defaultdict
import os
import numpy as np
from tabulate import tabulate
from tqdm import tqdm
from torch.utils.data import DataLoader
from oslactionspotting.core.utils.io import load_json, store_gz_json, store_json

# from oslactionspotting.core.utils.score import compute_mAPs_E2E

import os
import sys
from collections import defaultdict
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt


def parse_ground_truth(truth):
    """Parse ground truth dict to create dict with the following structure:
    {label : {game : list of frames index}}

    Args:
        truth : object containing list of informations about videos.

    Returns:
        label_dict : {label : {game : list of frames index}}.
    """
    label_dict = defaultdict(lambda: defaultdict(list))
    for x in truth:
        for e in x["events"]:
            label_dict[e["label"]][x["path"]].append(e["frame"])
    return label_dict


def get_predictions(pred, label=None):
    """Get a list of all predictions for a particular label.

    Args:
        pred : Object containing predictions data.

    Returns:
        flat_pred (List): List of tuples containing the name of the video, the frame index and
        the score confidence for each occurence of the particular label.
    """
    flat_pred = []
    for x in pred:
        for e in x["events"]:
            if label is None or e["label"] == label:
                flat_pred.append((x["video"], e["frame"], e["confidence"]))
    flat_pred.sort(key=lambda x: x[-1], reverse=True)
    return flat_pred


def compute_average_precision(
    pred,
    truth,
    tolerance=0,
    min_precision=0,
    plot_ax=None,
    plot_label=None,
    plot_raw_pr=True,
):
    """Compute average precision of predictions regarding truth data with a certain tolerance.

    Args:
        pred: prediction data.
        truth: groundtruth data.
        tolerance (int).
            Default: 0.
        min_precision (int).
            Default: 0.
        plot_ax (list): list of indexes for the axes.
            Default: None.
        plot_label (string): label of the plot.
            Default: None.
        plot_raw_pr (bool): whether to plot raw predictions.
            Default: False.
    Returns:
        average precision.
    """
    total = sum([len(x) for x in truth.values()])
    recalled = set()

    # The full precision curve has TOTAL number of bins, when recall increases
    # by in increments of one
    pc = []
    _prev_score = 1
    for i, (video, frame, score) in enumerate(pred, 1):
        assert score <= _prev_score
        _prev_score = score

        # Find the ground truth frame that is closest to the prediction
        gt_closest = None
        for gt_frame in truth.get(video, []):
            if (video, gt_frame) in recalled:
                continue
            if gt_closest is None or (abs(frame - gt_closest) > abs(frame - gt_frame)):
                gt_closest = gt_frame

        # Record precision each time a true positive is encountered
        if gt_closest is not None and abs(frame - gt_closest) <= tolerance:
            recalled.add((video, gt_closest))
            p = len(recalled) / i
            pc.append(p)

            # Stop evaluation early if the precision is too low.
            # Not used, however when nin_precision is 0.
            if p < min_precision:
                break

    interp_pc = []
    max_p = 0
    for p in pc[::-1]:
        max_p = max(p, max_p)
        interp_pc.append(max_p)
    interp_pc.reverse()  # Not actually necessary for integration

    if plot_ax is not None:
        rc = np.arange(1, len(pc) + 1) / total
        if plot_raw_pr:
            plot_ax.plot(rc, pc, label=plot_label, alpha=0.8)
        plot_ax.plot(rc, interp_pc, label=plot_label, alpha=0.8)

    # Compute AUC by integrating up to TOTAL bins
    return sum(interp_pc) / total


def compute_mAPs_E2E(truth, pred, tolerances=[0, 1, 2, 4], plot_pr=False):
    """Compute mAPs metric for the training module for the E2E method.

    Args:
        truth : Object containing the ground truth data.
        pred : Object containing the predictions data.
        tolerances (List[int]): List of tolerances values.
            Default: [0, 1, 2, 4].
        plot_pr (bool): Whether to plot or not the precision recall curve.
            Default: False.
    """

    assert {v["path"] for v in truth} == {
        v["video"] for v in pred
    }, "Video set mismatch!"

    truth_by_label = parse_ground_truth(truth)

    fig, axes = None, None
    if plot_pr:
        fig, axes = plt.subplots(
            len(truth_by_label),
            len(tolerances),
            sharex=True,
            sharey=True,
            figsize=(16, 16),
        )

    class_aps_for_tol = []
    mAPs = []
    for i, tol in enumerate(tolerances):
        class_aps = []
        for j, (label, truth_for_label) in enumerate(sorted(truth_by_label.items())):
            ap = compute_average_precision(
                get_predictions(pred, label=label),
                truth_for_label,
                tolerance=tol,
                plot_ax=axes[j, i] if axes is not None else None,
            )
            class_aps.append((label, ap))
        mAP = np.mean([x[1] for x in class_aps])
        mAPs.append(mAP)
        class_aps.append(("mAP", mAP))
        class_aps_for_tol.append(class_aps)
    header = ["AP @ tol"] + tolerances
    rows = []
    for c, _ in class_aps_for_tol[0]:
        row = [c]
        for class_aps in class_aps_for_tol:
            for c2, val in class_aps:
                if c2 == c:
                    row.append(val * 100)
        rows.append(row)
    print(tabulate(rows, headers=header, floatfmt="0.2f"))

    print("Avg mAP (across tolerances): {:0.2f}".format(np.mean(mAPs) * 100))

    if plot_pr:
        for i, tol in enumerate(tolerances):
            for j, label in enumerate(sorted(truth_by_label.keys())):
                ax = axes[j, i]
                ax.set_xlabel("Recall")
                ax.set_xlim(0, 1)
                ax.set_ylabel("Precision")
                ax.set_ylim(0, 1.01)
                ax.set_title("{} @ tol={}".format(label, tol))
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    sys.stdout.flush()
    return mAPs, tolerances


class ErrorStat:
    """Class to have error statistics"""

    def __init__(self):
        self._total = 0
        self._err = 0

    def update(self, true, pred):
        self._err += np.sum(true != pred)
        self._total += true.shape[0]

    def get(self):
        return self._err / self._total

    def get_acc(self):
        return 1.0 - self._get()


class ForegroundF1:
    """Class to have f1 scores"""

    def __init__(self):
        self._tp = defaultdict(int)
        self._fp = defaultdict(int)
        self._fn = defaultdict(int)

    def update(self, true, pred):
        if pred != 0:
            if true != 0:
                self._tp[None] += 1
            else:
                self._fp[None] += 1

            if pred == true:
                self._tp[pred] += 1
            else:
                self._fp[pred] += 1
                if true != 0:
                    self._fn[true] += 1
        elif true != 0:
            self._fn[None] += 1
            self._fn[true] += 1

    def get(self, k):
        return self._f1(k)

    def tp_fp_fn(self, k):
        return self._tp[k], self._fp[k], self._fn[k]

    def _f1(self, k):
        denom = self._tp[k] + 0.5 * self._fp[k] + 0.5 * self._fn[k]
        if denom == 0:
            assert self._tp[k] == 0
            denom = 1
        return self._tp[k] / denom


def process_frame_predictions(
    dali, dataset, classes, pred_dict, high_recall_score_threshold=0.01
):
    """Process predictions by computing statistics, creating dictionnaries
    with predictions and their associated informations

    Args:
        dali (bool): If data processing with dali or opencv.
        dataset (Dataset or DaliGenericIterator).
        classes (Dict): Classes associated with indexes.
        pred_dict (Dict): Mapping between clip and a tuple of scores and support.
        high_recall_score_threshold (float):
            Default: 0.01.

    Returns:
        err (ErrorStat).
        f1 (ForegroundF1).
        pred_events (List[dict]): List of dictionnaries with video, events, fps. Only one class maximum per frame.
        pred_events_high_recall (List[dict]): List of dictionnaries with video, events, fps. Several possible classes per frame.
        pred_scores (dict): Mapping between videos and associated scores.
    """
    classes_inv = {v: k for k, v in classes.items()}

    fps_dict = {}

    for video, _, fps in dataset.videos:
        fps_dict[video] = fps

    err = ErrorStat()
    f1 = ForegroundF1()

    pred_events = []
    pred_events_high_recall = []
    pred_scores = {}
    for video, (scores, support) in sorted(pred_dict.items()):
        label = dataset.get_labels(video)

        # support[support == 0] = 1   # get rid of divide by zero
        if dali:
            assert np.min(support[1:]) > 0, (video, support[1:].tolist())
            scores[1:] /= support[1:, None]
            pred = np.argmax(scores[1:], axis=1)
            err.update(label[1:], pred)
        else:
            assert np.min(support) > 0, (video, support.tolist())
            scores /= support[:, None]
            pred = np.argmax(scores, axis=1)
            err.update(label, pred)

        pred_scores[video] = scores.tolist()

        events = []
        events_high_recall = []
        # for i in range(1,pred.shape[0]):
        for i in range(pred.shape[0]):

            if dali:
                f1.update(label[i + 1], pred[i])
            else:
                f1.update(label[i], pred[i])
            if pred[i] != 0:
                if dali:
                    tmp = i + 1
                else:
                    tmp = i
                seconds = int((tmp // fps_dict[video]) % 60)
                minutes = int((tmp // fps_dict[video]) // 60)
                events.append(
                    {
                        "label": classes_inv[pred[i]],
                        "position": int((tmp * 1000) / fps_dict[video]),
                        "gameTime": f"{minutes:02.0f}:{seconds:02.0f}",
                        # 'frame': i,
                        "frame": tmp,
                        "confidence": scores[tmp, pred[i]].item(),
                        # 'score': scores[i, pred[i]].item()
                    }
                )

            for j in classes_inv:
                if dali:
                    tmp = i + 1
                else:
                    tmp = i
                if scores[tmp, j] >= high_recall_score_threshold:
                    # if scores[i, j] >= high_recall_score_threshold:
                    seconds = int((tmp // fps_dict[video]) % 60)
                    minutes = int((tmp // fps_dict[video]) // 60)
                    events_high_recall.append(
                        {
                            "label": classes_inv[j],
                            "position": int((tmp * 1000) / fps_dict[video]),
                            "gameTime": f"{minutes:02.0f}:{seconds:02.0f}",
                            "frame": tmp,
                            # 'frame': i,
                            "confidence": scores[tmp, j].item(),
                            # 'score': scores[i, j].item()
                        }
                    )
        pred_events.append({"video": video, "events": events, "fps": fps_dict[video]})
        pred_events_high_recall.append(
            {"video": video, "events": events_high_recall, "fps": fps_dict[video]}
        )

    return err, f1, pred_events, pred_events_high_recall, pred_scores


def infer_and_process_predictions_e2e(
    model,
    dali,
    dataset,
    split,
    classes,
    save_pred,
    calc_stats=True,
    dataloader_params=None,
    return_pred=False,
):
    """Infer prediction of actions from clips, process these predictions.

    Args:
        model .
        dali (bool): Whether dali has been used or opencv to process videos.
        dataset (Dataset or DaliGenericIterator).
        split (string): Split of the data.
        classes (dict) : Classes associated with indexes.
        save_pred (bool) : Save predictions or not.
        calc_stats (bool) : display stats or not.
            Default: True.
        dataloader_params (dict): Parameters for the dataloader.
            Default: None.
        return_pred (bool): Return dict of predictions or not.
            Default: False

    Returns:
        pred_events_high_recall (List[dict]): List of dictionnaries with video, events, fps. Several possible classes per frame.
        avg_mAP (float): Average mean AP computed for the predictions.
    """
    # print(dataset.)
    pred_dict = {}
    for video, video_len, _ in dataset.videos:
        pred_dict[video] = (
            np.zeros((video_len, len(classes) + 1), np.float32),
            np.zeros(video_len, np.int32),
        )

    batch_size = dataloader_params.batch_size

    for clip in tqdm(
        dataset
        if dali
        else DataLoader(
            dataset,
            num_workers=dataloader_params.num_workers,
            pin_memory=dataloader_params.pin_memory,
            batch_size=batch_size,
        )
    ):
        if batch_size > 1:
            # Batched by dataloader
            _, batch_pred_scores = model.predict(clip["frame"])
            for i in range(clip["frame"].shape[0]):
                video = clip["video"][i]
                scores, support = pred_dict[video]
                pred_scores = batch_pred_scores[i]
                start = clip["start"][i].item()
                if start < 0:
                    pred_scores = pred_scores[-start:, :]
                    start = 0
                end = start + pred_scores.shape[0]
                if end >= scores.shape[0]:
                    end = scores.shape[0]
                    pred_scores = pred_scores[: end - start, :]
                scores[start:end, :] += pred_scores
                support[start:end] += 1

        else:
            # Batched by dataset
            scores, support = pred_dict[clip["video"][0]]

            start = clip["start"][0].item()
            # start=start-1
            _, pred_scores = model.predict(clip["frame"][0])
            if start < 0:
                pred_scores = pred_scores[:, -start:, :]
                start = 0
            end = start + pred_scores.shape[1]
            if end >= scores.shape[0]:
                end = scores.shape[0]
                pred_scores = pred_scores[:, : end - start, :]

            scores[start:end, :] += np.sum(pred_scores, axis=0)
            support[start:end] += pred_scores.shape[0]

    err, f1, pred_events, pred_events_high_recall, pred_scores = (
        process_frame_predictions(dali, dataset, classes, pred_dict)
    )

    avg_mAP = None
    if calc_stats:
        print("=== Results on {} (w/o NMS) ===".format(split))
        print("Error (frame-level): {:0.2f}\n".format(err.get() * 100))

        def get_f1_tab_row(str_k):
            k = classes[str_k] if str_k != "any" else None
            return [str_k, f1.get(k) * 100, *f1.tp_fp_fn(k)]

        rows = [get_f1_tab_row("any")]
        for c in sorted(classes):
            rows.append(get_f1_tab_row(c))
        print(
            tabulate(
                rows, headers=["Exact frame", "F1", "TP", "FP", "FN"], floatfmt="0.2f"
            )
        )
        print()

        mAPs, _ = compute_mAPs_E2E(dataset.labels, pred_events_high_recall)
        avg_mAP = np.mean(mAPs[1:])

    if save_pred is not None:
        store_json(save_pred + ".json", pred_events, pretty=True)
        store_gz_json(save_pred + ".recall.json.gz", pred_events_high_recall)
        # if save_scores:
        #     store_gz_json(save_pred + '.score.json.gz', pred_scores)
    if return_pred:
        return pred_events_high_recall
    return avg_mAP


def search_best_epoch(work_dir):
    """
    Args:
        work_dir (string): Path in which there is the json file that contains losses for each epoch.

    Returns:
        epoch/epoch_mAP (int): The best epoch.
    """
    loss = load_json(os.path.join(work_dir, "loss.json"))
    valid_mAP = 0
    valid = float("inf")
    epoch = -1
    epoch_mAP = -1
    for epoch_loss in loss:
        if epoch_loss["valid_mAP"] > valid_mAP:
            valid_mAP = epoch_loss["valid_mAP"]
            epoch_mAP = epoch_loss["epoch"]
        if epoch_loss["valid"] < valid:
            valid = epoch_loss["valid"]
            epoch = epoch_loss["epoch"]
    if epoch_mAP != -1:
        return epoch_mAP
    else:
        return epoch


np.seterr(divide="ignore", invalid="ignore")
