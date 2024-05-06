from tabulate import tabulate
from OSLActionSpotting.core.utils.eval import compute_performances_mAP, get_closest_action_index, label2vector, non_maximum_supression, predictions2vector, store_eval_files_json
from OSLActionSpotting.core.utils.io import load_gz_json, load_text, load_json

import logging
from SoccerNet.Evaluation.ActionSpotting import evaluate

import numpy as np

import json

import json
from tqdm import tqdm
import os

from SoccerNet.Evaluation.utils import INVERSE_EVENT_DICTIONARY_V2

def build_evaluator(cfg, default_args=None):
    """Build a evaluator from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        default_args (dict | None, optional): Default initialization arguments.
            Default: None.

    Returns:
        evaluator: The constructed evaluator.
    """
    if cfg.runner.type == "runner_JSON":
        evaluator = Evaluator(cfg=cfg,
                        evaluate_Spotting=evaluate_pred_JSON)
    elif cfg.runner.type == "runner_pooling":
        evaluator = Evaluator(cfg=cfg,
                        evaluate_Spotting=evaluate_pred_SN)
    elif cfg.runner.type == "runner_CALF":
        evaluator = Evaluator(cfg=cfg,
                        evaluate_Spotting=evaluate_pred_SN)
    elif cfg.runner.type == "runner_e2e":
        evaluator = Evaluator(cfg=cfg,
                        evaluate_Spotting=evaluate_pred_E2E)
    
    return evaluator


class Evaluator():
    def __init__(self, 
                 cfg, 
                evaluate_Spotting):
        self.cfg = cfg
        self.evaluate_Spotting = evaluate_Spotting
    
    def evaluate(self, cfg_testset):
        # results = os.path.join(self.cfg.work_dir,cfg_testset.results)
        
        self.evaluate_Spotting(cfg_testset, self.cfg.work_dir, cfg_testset.results, cfg_testset.metric)
        


def evaluate_pred_E2E(cfg, work_dir, pred_path,metric="loose"):
    # Params:
    #   - SoccerNet_path: path for labels (folder or zipped file)
    #   - Predictions_path: path for predictions (folder or zipped file)
    #   - prediction_file: name of the predicted files - if set to None, try to infer it
    #   - split: split to evaluate from ["test", "challenge"]
    #   - frame_rate: frame rate to evalaute from [2]
    # Return:
    #   - details mAP

    results = os.path.join(work_dir, pred_path)

    if os.path.isfile(results) and (results.endswith('.gz') or results.endswith('.json')):
        pred = (load_gz_json if results.endswith('.gz') else load_json)(
            results)
        nms_window = cfg.nms_window
        if isinstance(pred, list):
            if nms_window > 0:
                logging.info('Applying NMS: ' +  str(nms_window))
                pred = non_maximum_supression(pred, nms_window)

            only_one_file = store_eval_files_json(pred,os.path.join(work_dir, pred_path.split('.gz')[0].split('.json')[0]))
            logging.info('Done processing prediction files!')
            if only_one_file : results =  os.path.join(work_dir, pred_path.split('.gz')[0].split('.json')[0], 'results_spotting.json')
            else : results = os.path.join(work_dir, pred_path.split('.gz')[0].split('.json')[0])


    with open(cfg.path) as f :
        GT_data = json.load(f)
    
    pred_path_is_json = False
    if results.endswith('.json'):
        pred_path_is_json = True
        with open(results) as f :
            pred_data = json.load(f)

    targets_numpy = list()
    detections_numpy = list()
    closests_numpy = list()
    classes = load_text(cfg.classes)
    EVENT_DICTIONARY = {x: i  for i, x in enumerate(classes)}
    INVERSE_EVENT_DICTIONARY = {i: x  for i, x in enumerate(classes)}

    if isinstance(GT_data, list):
        videos = GT_data
    else:
        videos = [GT_data]

    for game in tqdm(videos):

        # fetch labels
        if "events" in game.keys():
            labels = game["events"]
        elif "annotations" in game.keys():
            labels = game["annotations"]

        # # convert labels to dense vector
        # dense_labels = label2vector_e2e(labels, game["num_frames_dali"],
        #     num_classes=len(classes), 
        #     EVENT_DICTIONARY=EVENT_DICTIONARY)
        
        if not pred_path_is_json:
            try:
                with open(os.path.join(results,game["video"].rsplit('_', 1)[0],'results_spotting.json')) as f :
                    pred_data = json.load(f)
                    # predictions = pred_data['predictions']
            except FileNotFoundError:
                continue
        predictions = pred_data['predictions']

        # convert labels to dense vector
        dense_labels = label2vector(labels, vector_size=game["num_frames"] if "num_frames" in game.keys() else None,
            num_classes=len(classes), 
            EVENT_DICTIONARY=EVENT_DICTIONARY, framerate=pred_data['fps'] if 'fps' in pred_data.keys() else  cfg.extract_fps)
        
        # convert predictions to vector
        dense_predictions = predictions2vector(predictions, vector_size=game["num_frames"] if "num_frames" in game.keys() else None,
                                               num_classes=len(classes),
                                               EVENT_DICTIONARY=EVENT_DICTIONARY)


        targets_numpy.append(dense_labels)
        detections_numpy.append(dense_predictions)

        closest_numpy = np.zeros(dense_labels.shape)-1
        #Get the closest action index
        closests_numpy.append(get_closest_action_index(dense_labels, closest_numpy))
        
    if targets_numpy :
        return compute_performances_mAP(metric, targets_numpy, detections_numpy, closests_numpy, INVERSE_EVENT_DICTIONARY)


def evaluate_pred_JSON(cfg, work_dir, pred_path, metric="loose"):
    # Params:
    #   - SoccerNet_path: path for labels (folder or zipped file)
    #   - Predictions_path: path for predictions (folder or zipped file)
    #   - prediction_file: name of the predicted files - if set to None, try to infer it
    #   - split: split to evaluate from ["test", "challenge"]
    #   - frame_rate: frame rate to evalaute from [2]
    # Return:
    #   - details mAP

    # challenge sets to be tested on EvalAI
    if "challenge" in cfg.split: 
        print("Visit eval.ai to evaluate performances on Challenge set")
        return None
    
    pred_path = os.path.join(work_dir, pred_path)
    with open(cfg.path) as f :
        print(cfg.path)
        GT_data = json.load(f)
    

    pred_path_is_json = False
    if pred_path.endswith('.json'):
        pred_path_is_json = True
        with open(pred_path) as f :
            print(pred_path)
            pred_data = json.load(f)

    targets_numpy = list()
    detections_numpy = list()
    closests_numpy = list()

    if 'labels' in GT_data.keys():
        EVENT_DICTIONARY = {cls: i_cls for i_cls, cls in enumerate(GT_data["labels"])}
        INVERSE_EVENT_DICTIONARY = {i_cls: cls for i_cls, cls in enumerate(GT_data["labels"])}
        num_classes = len(GT_data["labels"])
    else:
        classes = load_text(cfg.classes)
        EVENT_DICTIONARY = {x: i  for i, x in enumerate(classes)}
        INVERSE_EVENT_DICTIONARY = {i: x  for i, x in enumerate(classes)}
        num_classes = len(classes)

    if 'videos' in GT_data.keys():
        videos = GT_data["videos"]
    else:
        videos = [GT_data]

    for game in tqdm(videos):
        # fetch labels
        labels = game["annotations"]

        # convert labels to dense vector
        dense_labels = label2vector(labels, 
            num_classes=num_classes, 
            EVENT_DICTIONARY=EVENT_DICTIONARY)
        
        if not pred_path_is_json:
            try:
                with open(os.path.join(pred_path,os.path.splitext(game["path_features"])[0],'results_spotting.json')) as f :
                    pred_data = json.load(f)
                    # predictions = pred_data['predictions']
            except FileNotFoundError:
                continue
        predictions = pred_data['predictions']
        # for video in pred_data["videos"]:
        #     if video["path_features"] == game["path_features"]:
        #         predictions = video["predictions"]
        #         break

        # convert predictions to vector
        dense_predictions = predictions2vector(predictions,
            num_classes=num_classes,
            EVENT_DICTIONARY=EVENT_DICTIONARY)

        targets_numpy.append(dense_labels)
        detections_numpy.append(dense_predictions)

        closest_numpy = np.zeros(dense_labels.shape)-1

        #Get the closest action index
        closests_numpy.append(get_closest_action_index(dense_labels, closest_numpy))
        
    if targets_numpy :
        return compute_performances_mAP(metric, targets_numpy, detections_numpy, closests_numpy, INVERSE_EVENT_DICTIONARY)
    
def evaluate_pred_SN(cfg, work_dir, pred_path, metric="loose"):
    # Params:
    #   - SoccerNet_path: path for labels (folder or zipped file)
    #   - Predictions_path: path for predictions (folder or zipped file)
    #   - prediction_file: name of the predicted files - if set to None, try to infer it
    #   - split: split to evaluate from ["test", "challenge"]
    #   - frame_rate: frame rate to evalaute from [2]
    # Return:
    #   - details mAP

    # challenge sets to be tested on EvalAI
    if "challenge" in cfg.split: 
        print("Visit eval.ai to evaluate performances on Challenge set")
        return None
    # GT_path = cfg.data_root
    pred_path = os.path.join(work_dir, pred_path)
    results = evaluate(SoccerNet_path=cfg.data_root, 
                    Predictions_path=pred_path,
                    split=cfg.split,
                    prediction_file="results_spotting.json", 
                    version=cfg.version,
                    metric=metric)
    # results = {
    #     "a_mAP": a_mAP,
    #     "a_mAP_per_class": a_mAP_per_class,
    # }
    rows = []
    for i in range(len(results['a_mAP_per_class'])):
        label = INVERSE_EVENT_DICTIONARY_V2[i]
        rows.append((
            label,
            '{:0.2f}'.format(results['a_mAP_per_class'][i] * 100),
            '{:0.2f}'.format(results['a_mAP_per_class_visible'][i] * 100),
            '{:0.2f}'.format(results['a_mAP_per_class_unshown'][i] * 100)
        ))
    rows.append((
        'Average mAP',
        '{:0.2f}'.format(results['a_mAP'] * 100),
        '{:0.2f}'.format(results['a_mAP_visible'] * 100),
        '{:0.2f}'.format(results['a_mAP_unshown'] * 100)
    ))

    logging.info("Best Performance at end of training ")
    logging.info('Metric: ' +  metric)
    print(tabulate(rows, headers=['', 'Any', 'Visible', 'Unseen']))
    # logging.info("a_mAP visibility all: " +  str(results["a_mAP"]))
    # logging.info("a_mAP visibility all per class: " +  str( results["a_mAP_per_class"]))
    
    return results