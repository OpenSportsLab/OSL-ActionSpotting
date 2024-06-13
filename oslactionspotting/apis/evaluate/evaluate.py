from tabulate import tabulate
from oslactionspotting.apis.evaluate.utils import (
    compute_performances_mAP,
    get_closest_action_index,
    label2vector,
    non_maximum_supression,
    predictions2vector,
    store_eval_files_json,
)
from oslactionspotting.core.utils.io import load_gz_json, load_text, load_json

import logging
from SoccerNet.Evaluation.ActionSpotting import evaluate

import numpy as np

import json

import json
from tqdm import tqdm
import os

from SoccerNet.Evaluation.utils import INVERSE_EVENT_DICTIONARY_V2


class Evaluator:
    """Evaluator class that is used to make easier the process of evaluate since there is only
    one evaluate method that uses the evaluate_Spotting method.

    Args:
        cfg (dict): Config dict.
        evaluate_Spotting (method): The method that is used to evaluate.
    """

    def __init__(self, cfg, evaluate_Spotting):
        self.cfg = cfg
        self.evaluate_Spotting = evaluate_Spotting

    def evaluate(self, cfg_testset):
        """Evaluate predictions.

        Args:
            cfg_testset (dict): Config dict that contains informations for the predictions.
        """
        self.evaluate_Spotting(
            cfg_testset, self.cfg.work_dir, cfg_testset.results, cfg_testset.metric
        )


def evaluate_common_JSON(cfg, results, metric):
    if cfg.path == None:
        return
    with open(cfg.path) as f:
        GT_data = json.load(f)

    pred_path_is_json = False
    if results.endswith(".json"):
        pred_path_is_json = True
        with open(results) as f:
            pred_data = json.load(f)

    targets_numpy = list()
    detections_numpy = list()
    closests_numpy = list()

    if "labels" in GT_data.keys():
        classes = GT_data["labels"]
    else:
        assert isinstance(cfg.classes, list) or os.path.isfile(cfg.classes)
        if isinstance(cfg.classes, list):
            classes = cfg.classes
        else:
            classes = load_text(cfg.classes)

    classes = sorted(classes)
    EVENT_DICTIONARY = {x: i for i, x in enumerate(classes)}
    INVERSE_EVENT_DICTIONARY = {i: x for i, x in enumerate(classes)}

    if "videos" in GT_data.keys():
        videos = GT_data["videos"]
    else:
        videos = [GT_data]

    for game in tqdm(videos):

        # fetch labels
        labels = game["annotations"]
        if not pred_path_is_json:
            try:
                with open(
                    os.path.join(
                        results,
                        os.path.splitext(game["path"])[0],
                        "results_spotting.json",
                    )
                ) as f:
                    pred_data = json.load(f)
            except FileNotFoundError:
                continue
        predictions = pred_data["predictions"]

        # convert labels to dense vector
        dense_labels = label2vector(
            labels,
            num_classes=len(classes),
            EVENT_DICTIONARY=EVENT_DICTIONARY,
            framerate=(
                pred_data["fps"] if "fps" in pred_data.keys() else cfg.extract_fps
            ),
        )

        # convert predictions to vector
        dense_predictions = predictions2vector(
            predictions,
            vector_size=game["num_frames"] if "num_frames" in game.keys() else None,
            framerate=(
                pred_data["fps"] if "fps" in pred_data.keys() else cfg.extract_fps
            ),
            num_classes=len(classes),
            EVENT_DICTIONARY=EVENT_DICTIONARY,
        )

        targets_numpy.append(dense_labels)
        detections_numpy.append(dense_predictions)

        closest_numpy = np.zeros(dense_labels.shape) - 1
        # Get the closest action index
        closests_numpy.append(get_closest_action_index(dense_labels, closest_numpy))

    if targets_numpy:
        return compute_performances_mAP(
            metric,
            targets_numpy,
            detections_numpy,
            closests_numpy,
            INVERSE_EVENT_DICTIONARY,
        )
    else:
        return


def evaluate_pred_E2E(cfg, work_dir, pred_path, metric="loose"):
    """Evaluate predictions infered with E2E method and display performances.
    Args:
        cfg (dict): It should containt the keys; classes (list of classes), path (path of the groundtruth data).
        It should contain the key nms_window if evaluation of raw predictions. It should containt the key extract_fps if predictions file do not contain the fps at which the frames have been processed to infer.
        work_dir: The folder path under which the prediction files are stored.
        pred_path: The path for predictions files. It can be:
            - folder path (that contains predictions files)
            - file path (if raw prediction that needs to be processed first)
        metric (string): metric used to evaluate.
            In ["loose","tight","at1","at2","at3","at4","at5"].
            Default: "loose".

    Returns
        The different mAPs computed.
    """

    results = os.path.join(work_dir, pred_path)

    if os.path.isfile(results) and (
        results.endswith(".gz") or results.endswith(".json")
    ):
        pred = (load_gz_json if results.endswith(".gz") else load_json)(results)
        nms_window = cfg.nms_window
        if isinstance(pred, list):
            if nms_window > 0:
                logging.info("Applying NMS: " + str(nms_window))
                pred = non_maximum_supression(pred, nms_window)

            only_one_file = store_eval_files_json(
                pred,
                os.path.join(work_dir, pred_path.split(".gz")[0].split(".json")[0]),
            )
            logging.info("Done processing prediction files!")
            if only_one_file:
                results = os.path.join(
                    work_dir,
                    pred_path.split(".gz")[0].split(".json")[0],
                    "results_spotting.json",
                )
            else:
                results = os.path.join(
                    work_dir, pred_path.split(".gz")[0].split(".json")[0]
                )
    return evaluate_common_JSON(cfg, results, metric)


def evaluate_pred_JSON(cfg, work_dir, pred_path, metric="loose"):
    """Evaluate predictions infered with Json files and display performances.
    Args:
        cfg (dict): It should containt the key path (path of the groundtruth data). It should containt the key classes (list of classes) if the different classes are not in the ground truth data.
        work_dir: The folder path under which the prediction files are stored.
        pred_path: The path for predictions files. It can be:
            - folder path (that contains predictions files)
            - json file path if evaluate only one json file.
        metric (string): metric used to evaluate.
            In ["loose","tight","at1","at2","at3","at4","at5"].
            Default: "loose".

    Returns
        The different mAPs computed.
    """
    return evaluate_common_JSON(cfg, os.path.join(work_dir, pred_path), metric)


def evaluate_pred_SN(cfg, work_dir, pred_path, metric="loose"):
    """Evaluate predictions infered using SoccerNetv2 splits and display performances. This method should be used only for SoccerNetv2 dataset.
    Args:
        cfg (dict): It should containt the key path (path of the groundtruth data). It should containt the key classes (list of classes) if the different classes are not in the ground truth data.
        work_dir: The folder path under which the prediction files are stored.
        pred_path: The path for predictions files.
        metric (string): metric used to evaluate.
            In ["loose","tight","at1","at2","at3","at4","at5"].
            Default: "loose".

    Returns
        The different mAPs computed.
    """

    # challenge sets to be tested on EvalAI
    if "challenge" in cfg.split:
        print("Visit eval.ai to evaluate performances on Challenge set")
        return None
    # GT_path = cfg.data_root
    pred_path = os.path.join(work_dir, pred_path)
    results = evaluate(
        SoccerNet_path=cfg.data_root,
        Predictions_path=pred_path,
        split=cfg.split,
        prediction_file="results_spotting.json",
        version=getattr(cfg, "version", 2),
        metric=metric,
    )
    rows = []
    for i in range(len(results["a_mAP_per_class"])):
        label = INVERSE_EVENT_DICTIONARY_V2[i]
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
    print(tabulate(rows, headers=["", "Any", "Visible", "Unseen"]))
    # logging.info("a_mAP visibility all: " +  str(results["a_mAP"]))
    # logging.info("a_mAP visibility all per class: " +  str( results["a_mAP_per_class"]))

    return results
