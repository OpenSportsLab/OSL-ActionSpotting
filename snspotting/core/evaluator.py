import torch
import os

from snspotting.core.utils import CustomProgressBar

from .runner import build_runner
from snspotting.datasets import build_dataset, build_dataloader

import logging
from SoccerNet.Evaluation.ActionSpotting import evaluate


import os
import numpy as np

from SoccerNet.utils import getListGames


import json

import json
import zipfile
from tqdm import tqdm


import glob

import pytorch_lightning as pl


def build_evaluator(cfg, model, default_args=None):
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
                        evaluate_Spotting=evaluate_JSON,
                        model=model)
    elif cfg.runner.type == "runner_pooling":
        evaluator = Evaluator(cfg=cfg,
                        evaluate_Spotting=evaluate_SN,
                        model=model)
    elif cfg.runner.type == "runner_CALF":
        evaluator = Evaluator(cfg=cfg,
                        evaluate_Spotting=evaluate_SN,
                        model=model)
    
    return evaluator


class Evaluator():
    def __init__(self, cfg, 
                evaluate_Spotting,
                model):
        self.cfg = cfg
        self.model = model
        # self.runner = build_runner(cfg.runner, model)
        self.evaluate_Spotting = evaluate_Spotting

    def evaluate(self, cfg_testset):

        # Loop over dataset to evaluate
        splits_to_evaluate = cfg_testset.split
        for split in splits_to_evaluate:
            logging.info('split is %s',split)    
            cfg_testset.split = [split]

            # Build Dataset
            dataset_Test = build_dataset(cfg_testset,self.cfg.training.GPU)

            # Build Dataloader
            test_loader = build_dataloader(dataset_Test, cfg_testset.dataloader,self.cfg.training.GPU)

            # Run Inference on Dataset
            results = self.runner.infer_dataset(self.cfg, test_loader, self.model, overwrite=True)
            # evaluator = pl.Trainer(callbacks=[CustomProgressBar()],devices=[0],num_sanity_val_steps=0)
            # evaluator.predict(self.model,test_loader)
            # results = self.model.output_results
            # extract performances from results
            performances = self.evaluate_Spotting(cfg_testset, results)

            return performances



def evaluate_JSON(cfg, pred_path, metric="loose"):
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
    
    
    with open(cfg.path) as f :
        GT_data = json.load(f)
    with open(pred_path) as f :
        pred_data = json.load(f)

    targets_numpy = list()
    detections_numpy = list()
    closests_numpy = list()

    EVENT_DICTIONARY = {cls: i_cls for i_cls, cls in enumerate(GT_data["labels"])}

    for game in tqdm(GT_data["videos"]):

        # fetch labels
        labels = game["annotations"]

        # convert labels to dense vector
        dense_labels = label2vector(labels, 
            num_classes=len(GT_data["labels"]), 
            EVENT_DICTIONARY=EVENT_DICTIONARY)
        

        for video in pred_data["videos"]:
            if video["path_features"] == game["path_features"]:
                predictions = video["predictions"]
                break


        # convert predictions to vector
        dense_predictions = predictions2vector(predictions,
            num_classes=len(GT_data["labels"]),
            EVENT_DICTIONARY=EVENT_DICTIONARY)

        targets_numpy.append(dense_labels)
        detections_numpy.append(dense_predictions)

        closest_numpy = np.zeros(dense_labels.shape)-1
        #Get the closest action index
        for c in np.arange(dense_labels.shape[-1]):
            indexes = np.where(dense_labels[:,c] != 0)[0].tolist()
            if len(indexes) == 0 :
                continue
            indexes.insert(0,-indexes[0])
            indexes.append(2*closest_numpy.shape[0])
            for i in np.arange(len(indexes)-2)+1:
                start = max(0,(indexes[i-1]+indexes[i])//2)
                stop = min(closest_numpy.shape[0], (indexes[i]+indexes[i+1])//2)
                closest_numpy[start:stop,c] = dense_labels[indexes[i],c]
        closests_numpy.append(closest_numpy)


    if metric == "loose": deltas=np.arange(12)*5 + 5
    elif metric == "tight": deltas=np.arange(5)*1 + 1
    elif metric == "at1": deltas=np.array([1])
    elif metric == "at2": deltas=np.array([2]) 
    elif metric == "at3": deltas=np.array([3]) 
    elif metric == "at4": deltas=np.array([4]) 
    elif metric == "at5": deltas=np.array([5]) 

    # Compute the performances
    a_mAP, a_mAP_per_class = average_mAP(targets_numpy, 
    detections_numpy, closests_numpy,
    framerate=2, deltas=deltas)
    
    results = {
        "a_mAP": a_mAP,
        "a_mAP_per_class": a_mAP_per_class,
    }

    logging.info("Best Performance at end of training ")
    logging.info("a_mAP visibility all: " +  str(a_mAP))
    logging.info("a_mAP visibility all per class: " +  str( a_mAP_per_class))
    
    return results




def evaluate_SN(cfg, pred_path, metric="loose"):
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
    results = evaluate(SoccerNet_path=cfg.data_root, 
                    Predictions_path=pred_path,
                    split=cfg.split,
                    prediction_file="results_spotting.json", 
                    version=cfg.version)
    # results = {
    #     "a_mAP": a_mAP,
    #     "a_mAP_per_class": a_mAP_per_class,
    # }

    logging.info("Best Performance at end of training ")
    logging.info("a_mAP visibility all: " +  str(results["a_mAP"]))
    logging.info("a_mAP visibility all per class: " +  str( results["a_mAP_per_class"]))
    
    return results







def label2vector(labels, num_classes=17, framerate=2, version=2, EVENT_DICTIONARY={}):

    vector_size = 90*60*framerate

    dense_labels = np.zeros((vector_size, num_classes))

    for annotation in labels:

        time = annotation["gameTime"]
        event = annotation["label"]

        half = int(time[0])

        minutes = int(time[-5:-3])
        seconds = int(time[-2::])
        # annotation at millisecond precision
        if "position" in annotation:
            frame = int(framerate * ( int(annotation["position"])/1000 )) 
        # annotation at second precision
        else:
            frame = framerate * ( seconds + 60 * minutes ) 

        label = EVENT_DICTIONARY[event]

        frame = min(frame, vector_size-1)
        dense_labels[frame][label] = 1


    return dense_labels

def predictions2vector(predictions, num_classes=17, version=2, framerate=2, EVENT_DICTIONARY={}):

    vector_size = 90*60*framerate

    dense_predictions = np.zeros((vector_size, num_classes))-1

    for annotation in predictions:

        time = int(annotation["position"])
        event = annotation["label"]

        half = int(annotation["half"])

        frame = int(framerate * ( time/1000 ))

        label = EVENT_DICTIONARY[event]

        frame = min(frame, vector_size-1)
        dense_predictions[frame][label] = annotation["confidence"]

    return dense_predictions 


import numpy as np
from tqdm import tqdm
import time
np.seterr(divide='ignore', invalid='ignore')

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

    # Retrieving the important variables
    gt_indexes = np.where(target != 0)[0]
    gt_indexes_visible = np.where(target > 0)[0]
    gt_indexes_unshown = np.where(target < 0)[0]
    pred_indexes = np.where(detection >= 0)[0]
    pred_scores = detection[pred_indexes]

    # Array to save the results, each is [pred_scor,{1 or 0}]
    game_detections = np.zeros((len(pred_indexes),3))
    game_detections[:,0] = np.copy(pred_scores)
    game_detections[:,2] = np.copy(closest[pred_indexes])


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

            if abs(pred_index-gt_index) <= delta/2 and pred_score > max_score and pred_index not in remove_indexes:
                max_score = pred_score
                max_index = pred_index
                selected_game_index = game_index
            game_index += 1

        if max_index is not None:
            game_detections[selected_game_index,1]=1
            remove_indexes.append(max_index)

    return game_detections, len(gt_indexes_visible), len(gt_indexes_unshown)



def compute_precision_recall_curve(targets, closests, detections, delta):
    
    # Store the number of classes
    num_classes = targets[0].shape[-1]

    # 200 confidence thresholds between [0,1]
    thresholds = np.linspace(0,1,200)

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
        total_detections =  np.zeros((1, 3))
        total_detections[0,0] = -1
        n_gt_labels_visible = 0
        n_gt_labels_unshown = 0
        
        # Get the confidence scores and their corresponding TP or FP characteristics for each game
        for target, closest, detection in zip(targets, closests, detections):
            tmp_detections, tmp_n_gt_labels_visible, tmp_n_gt_labels_unshown = compute_class_scores(target[:,c], closest[:,c], detection[:,c], delta)
            total_detections = np.append(total_detections,tmp_detections,axis=0)
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
        total_detections_visible[np.where(total_detections_visible[:,2] <= 0.5)[0],0] = -1
        total_detections_unshown[np.where(total_detections_unshown[:,2] >= -0.5)[0],0] = -1

        # Get the precision and recall for each confidence threshold
        for threshold in thresholds:
            pred_indexes = np.where(total_detections[:,0]>=threshold)[0]
            pred_indexes_visible = np.where(total_detections_visible[:,0]>=threshold)[0]
            pred_indexes_unshown = np.where(total_detections_unshown[:,0]>=threshold)[0]
            TP = np.sum(total_detections[pred_indexes,1])
            TP_visible = np.sum(total_detections[pred_indexes_visible,1])
            TP_unshown = np.sum(total_detections[pred_indexes_unshown,1])
            p = np.nan_to_num(TP/len(pred_indexes))
            r = np.nan_to_num(TP/(n_gt_labels_visible + n_gt_labels_unshown))
            p_visible = np.nan_to_num(TP_visible/len(pred_indexes_visible))
            r_visible = np.nan_to_num(TP_visible/n_gt_labels_visible)
            p_unshown = np.nan_to_num(TP_unshown/len(pred_indexes_unshown))
            r_unshown = np.nan_to_num(TP_unshown/n_gt_labels_unshown)
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
        index_sort = np.argsort(recall[:,i])
        precision[:,i] = precision[index_sort,i]
        recall[:,i] = recall[index_sort,i]

    # Sort the points based on the recall, class per class
    for i in np.arange(num_classes):
        index_sort = np.argsort(recall_visible[:,i])
        precision_visible[:,i] = precision_visible[index_sort,i]
        recall_visible[:,i] = recall_visible[index_sort,i]

    # Sort the points based on the recall, class per class
    for i in np.arange(num_classes):
        index_sort = np.argsort(recall_unshown[:,i])
        precision_unshown[:,i] = precision_unshown[index_sort,i]
        recall_unshown[:,i] = recall_unshown[index_sort,i]

    return precision, recall, precision_visible, recall_visible, precision_unshown, recall_unshown

def compute_mAP(precision, recall):

    # Array for storing the AP per class
    AP = np.array([0.0]*precision.shape[-1])

    # Loop for all classes
    for i in np.arange(precision.shape[-1]):

        # 11 point interpolation
        for j in np.arange(11)/10:

            index_recall = np.where(recall[:,i] >= j)[0]

            possible_value_precision = precision[index_recall,i]
            max_value_precision = 0

            if possible_value_precision.shape[0] != 0:
                max_value_precision = np.max(possible_value_precision)

            AP[i] += max_value_precision

    mAP_per_class = AP/11

    return np.mean(mAP_per_class), mAP_per_class

# Tight: (SNv3): np.arange(5)*1 + 1
# Loose: (SNv1/v2): np.arange(12)*5 + 5
def delta_curve(targets, closests, detections,  framerate, deltas=np.arange(5)*1 + 1):

    mAP = list()
    mAP_per_class = list()
    mAP_visible = list()
    mAP_per_class_visible = list()
    mAP_unshown = list()
    mAP_per_class_unshown = list()

    for delta in tqdm(deltas*framerate):

        precision, recall, precision_visible, recall_visible, precision_unshown, recall_unshown = compute_precision_recall_curve(targets, closests, detections, delta)


        tmp_mAP, tmp_mAP_per_class = compute_mAP(precision, recall)
        mAP.append(tmp_mAP)
        mAP_per_class.append(tmp_mAP_per_class)
        # TODO: compute visible/undshown from another JSON file containing only the visible/unshown annotations
        # tmp_mAP_visible, tmp_mAP_per_class_visible = compute_mAP(precision_visible, recall_visible)
        # mAP_visible.append(tmp_mAP_visible)
        # mAP_per_class_visible.append(tmp_mAP_per_class_visible)
        # tmp_mAP_unshown, tmp_mAP_per_class_unshown = compute_mAP(precision_unshown, recall_unshown)
        # mAP_unshown.append(tmp_mAP_unshown)
        # mAP_per_class_unshown.append(tmp_mAP_per_class_unshown)

    return mAP, mAP_per_class


def average_mAP(targets, detections, closests, framerate=2, deltas=np.arange(5)*1 + 1):


    mAP, mAP_per_class = delta_curve(targets, closests, detections, framerate, deltas)
    
    if len(mAP) == 1:
        return mAP[0], mAP_per_class[0], mAP_visible[0], mAP_per_class_visible[0], mAP_unshown[0], mAP_per_class_unshown[0]
    
    # Compute the average mAP
    integral = 0.0
    for i in np.arange(len(mAP)-1):
        integral += (mAP[i]+mAP[i+1])/2
    a_mAP = integral/((len(mAP)-1))

    # integral_visible = 0.0
    # for i in np.arange(len(mAP_visible)-1):
    #     integral_visible += (mAP_visible[i]+mAP_visible[i+1])/2
    # a_mAP_visible = integral_visible/((len(mAP_visible)-1))

    # integral_unshown = 0.0
    # for i in np.arange(len(mAP_unshown)-1):
    #     integral_unshown += (mAP_unshown[i]+mAP_unshown[i+1])/2
    # a_mAP_unshown = integral_unshown/((len(mAP_unshown)-1))
    # a_mAP_unshown = a_mAP_unshown*17/13

    a_mAP_per_class = list()
    for c in np.arange(len(mAP_per_class[0])):
        integral_per_class = 0.0
        for i in np.arange(len(mAP_per_class)-1):
            integral_per_class += (mAP_per_class[i][c]+mAP_per_class[i+1][c])/2
        a_mAP_per_class.append(integral_per_class/((len(mAP_per_class)-1)))

    # a_mAP_per_class_visible = list()
    # for c in np.arange(len(mAP_per_class_visible[0])):
    #     integral_per_class_visible = 0.0
    #     for i in np.arange(len(mAP_per_class_visible)-1):
    #         integral_per_class_visible += (mAP_per_class_visible[i][c]+mAP_per_class_visible[i+1][c])/2
    #     a_mAP_per_class_visible.append(integral_per_class_visible/((len(mAP_per_class_visible)-1)))

    # a_mAP_per_class_unshown = list()
    # for c in np.arange(len(mAP_per_class_unshown[0])):
    #     integral_per_class_unshown = 0.0
    #     for i in np.arange(len(mAP_per_class_unshown)-1):
    #         integral_per_class_unshown += (mAP_per_class_unshown[i][c]+mAP_per_class_unshown[i+1][c])/2
    #     a_mAP_per_class_unshown.append(integral_per_class_unshown/((len(mAP_per_class_unshown)-1)))

    return a_mAP, a_mAP_per_class #, a_mAP_visible, a_mAP_per_class_visible, a_mAP_unshown, a_mAP_per_class_unshown



def LoadJsonFromZip(zippedFile, JsonPath):
    with zipfile.ZipFile(zippedFile, "r") as z:
        # print(filename)
        with z.open(JsonPath) as f:
            data = f.read()
            d = json.loads(data.decode("utf-8"))

    return d


