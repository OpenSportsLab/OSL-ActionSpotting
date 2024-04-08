from tabulate import tabulate
from snspotting.core.utils.eval import average_mAP, compute_performances_mAP, evaluate_e2e, get_closest_action_index, label2vector, non_maximum_supression, predictions2vector, store_eval_files_json
from snspotting.core.utils.io import load_gz_json, load_text
from snspotting.core.utils.lightning import CustomProgressBar

from snspotting.datasets import build_dataset, build_dataloader
import logging
from SoccerNet.Evaluation.ActionSpotting import evaluate

import numpy as np

import json

import json
from tqdm import tqdm
import os
import pytorch_lightning as pl

from SoccerNet.Evaluation.utils import INVERSE_EVENT_DICTIONARY_V2

from tools.parse_soccernet import load_json



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
    elif cfg.runner.type == "runner_e2e":
        evaluator = Evaluator(cfg=cfg,
                        evaluate_Spotting=evaluate_E2E,
                        model=model)
    
    return evaluator


class Evaluator():
    def __init__(self, cfg, 
                evaluate_Spotting,
                model):
        self.cfg = cfg

        self.dali = False
        if 'dali' in self.cfg.keys():self.dali = True
        self.model = model
        self.evaluate_Spotting = evaluate_Spotting

    def infer(self, cfg_testset):

        # Loop over dataset to evaluate
        splits_to_evaluate = cfg_testset.split
        print(splits_to_evaluate)
        for split in splits_to_evaluate:
            logging.info('split is %s',split)    
            cfg_testset.split = [split]

            # Build Dataset
            dataset_test = build_dataset(cfg_testset,self.cfg.training.GPU if 'GPU' in self.cfg.training.keys() else None, 
                                         {"classes": self.cfg.classes, 'repartitions' : self.cfg.repartitions}  if self.dali else None)

            # Build Dataloader
            if self.dali:
                test_loader = dataset_test
                test_loader.print_info()
            else:
                test_loader = build_dataloader(dataset_test, cfg_testset.dataloader,self.cfg.training.GPU)

            if self.cfg.model.type == 'E2E':
                pred_file = None
                if self.cfg.work_dir is not None:
                    pred_file = os.path.join(
                        self.cfg.work_dir, 'evaluate-{}'.format('test'))

                results = self.infer_e2e(self.model, self.dali, test_loader, split.upper(), self.cfg.classes, pred_file,
                            calc_stats=False)
            else:
                # Run Inference on Dataset
                evaluator = pl.Trainer(callbacks=[CustomProgressBar()],devices=[self.cfg.training.GPU],num_sanity_val_steps=0)
                evaluator.predict(self.model,test_loader)
                results = self.model.target_dir
                print(results)
            return results
    def infer_e2e(self, model, dali, dataset, split, classes, save_pred, calc_stats=True,
             save_scores=True):
        # pred_dict = {}
        # for video, video_len, _ in dataset.videos:
        #     pred_dict[video] = (
        #         np.zeros((video_len, len(classes) + 1), np.float32),
        #         np.zeros(video_len, np.int32))

        # # Do not up the batch size if the dataset augments
        # batch_size = 1 if dataset.augment else 4

        # for clip in tqdm(dataset if dali else DataLoader(
        #         dataset, num_workers=8, pin_memory=True,
        #         batch_size=batch_size
        # )):
        #     if batch_size > 1:
        #         # Batched by dataloader
        #         _, batch_pred_scores = model.predict(clip['frame'])
        #         for i in range(clip['frame'].shape[0]):
        #             video = clip['video'][i]
        #             scores, support = pred_dict[video]
        #             pred_scores = batch_pred_scores[i]
        #             start = clip['start'][i].item()
        #             if start < 0:
        #                 pred_scores = pred_scores[-start:, :]
        #                 start = 0
        #             end = start + pred_scores.shape[0]
        #             if end >= scores.shape[0]:
        #                 end = scores.shape[0]
        #                 pred_scores = pred_scores[:end - start, :]
        #             scores[start:end, :] += pred_scores
        #             support[start:end] += 1

        #     else:
        #         # Batched by dataset
        #         scores, support = pred_dict[clip['video'][0]]

        #         start = clip['start'][0].item()
        #         start=start-1
        #         _, pred_scores = model.predict(clip['frame'][0])
        #         if start < 0:
        #             pred_scores = pred_scores[:, -start:, :]
        #             start = 0
        #         end = start + pred_scores.shape[1]
        #         if end >= scores.shape[0]:
        #             end = scores.shape[0]
        #             pred_scores = pred_scores[:,:end - start, :]

        #         print(pred_scores.shape)
        #         scores[start:end, :] += np.sum(pred_scores, axis=0)
        #         support[start:end] += pred_scores.shape[0]

        # err, f1, pred_events, pred_events_high_recall, pred_scores = \
        #     process_frame_predictions(dataset, classes, pred_dict)
        
        # avg_mAP = None
        # if calc_stats:
        #     print('=== Results on {} (w/o NMS) ==='.format(split))
        #     print('Error (frame-level): {:0.2f}\n'.format(err.get() * 100))

        #     def get_f1_tab_row(str_k):
        #         k = classes[str_k] if str_k != 'any' else None
        #         return [str_k, f1.get(k) * 100, *f1.tp_fp_fn(k)]
        #     rows = [get_f1_tab_row('any')]
        #     for c in sorted(classes):
        #         rows.append(get_f1_tab_row(c))
        #     print(tabulate(rows, headers=['Exact frame', 'F1', 'TP', 'FP', 'FN'],
        #                 floatfmt='0.2f'))
        #     print()

        #     mAPs, _ = compute_mAPs(dataset.labels, pred_events_high_recall)
        #     avg_mAP = np.mean(mAPs[1:])

        # if save_pred is not None:
        #     store_json(save_pred + '.json', pred_events)
        #     store_gz_json(save_pred + '.recall.json.gz', pred_events_high_recall)
        #     if save_scores:
        #         store_gz_json(save_pred + '.score.json.gz', pred_scores)

        evaluate_e2e(model, dali, dataset, split, classes, save_pred, calc_stats, save_scores)
        pred_file = save_pred + '.recall.json.gz'
        return pred_file
        pred = (load_gz_json if pred_file.endswith('.gz') else load_json)(
                pred_file)
        nms_window = 2
        if nms_window > 0:
            logging.info('Applying NMS: ' +  str(nms_window))
            pred = non_maximum_supression(pred, nms_window)

        store_eval_files_json(pred,os.path.join(self.cfg.work_dir,'results_spotting_test'))
        logging.info('Done processing prediction files!')

        return os.path.join(self.cfg.work_dir,'results_spotting_test')
    def evaluate(self, cfg_testset, results):
        if results.endswith('.gz') or results.endswith('.json'):
            pred = (load_gz_json if results.endswith('.gz') else load_json)(
                results)
            nms_window = 2
            if nms_window > 0:
                logging.info('Applying NMS: ' +  str(nms_window))
                pred = non_maximum_supression(pred, nms_window)

            store_eval_files_json(pred,os.path.join(self.cfg.work_dir,'results_spotting_test'))
            logging.info('Done processing prediction files!')
            results = os.path.join(self.cfg.work_dir,'results_spotting_test')
        performances = self.evaluate_Spotting(cfg_testset, results, cfg_testset.metric)
        return performances


def evaluate_E2E(cfg, pred_path,metric="loose"):
    # Params:
    #   - SoccerNet_path: path for labels (folder or zipped file)
    #   - Predictions_path: path for predictions (folder or zipped file)
    #   - prediction_file: name of the predicted files - if set to None, try to infer it
    #   - split: split to evaluate from ["test", "challenge"]
    #   - frame_rate: frame rate to evalaute from [2]
    # Return:
    #   - details mAP

    
    with open(cfg.label_file) as f :
        print(cfg.label_file)
        GT_data = json.load(f)
    

    targets_numpy = list()
    detections_numpy = list()
    closests_numpy = list()
    classes = load_text(cfg.classes)
    EVENT_DICTIONARY = {x: i  for i, x in enumerate(classes)}
    INVERSE_EVENT_DICTIONARY = {i: x  for i, x in enumerate(classes)}
    for game in tqdm(GT_data):

        # fetch labels
        labels = game["events"]

        # convert labels to dense vector
        dense_labels = label2vector(labels, vector_size=game["num_frames_dali"],
            num_classes=len(classes), 
            EVENT_DICTIONARY=EVENT_DICTIONARY)
        # # convert labels to dense vector
        # dense_labels = label2vector_e2e(labels, game["num_frames_dali"],
        #     num_classes=len(classes), 
        #     EVENT_DICTIONARY=EVENT_DICTIONARY)
        
        with open(os.path.join(pred_path,game["video"].rsplit('_', 1)[0],'results_spotting.json')) as f :
            pred_data = json.load(f)
            predictions = pred_data['predictions']

        # convert predictions to vector
        dense_predictions = predictions2vector(predictions, vector_size=game["num_frames_dali"]
,            num_classes=len(classes),
            EVENT_DICTIONARY=EVENT_DICTIONARY)
#         # convert predictions to vector
#         dense_predictions = predictions2vector_e2e(predictions, game["num_frames_dali"]
# ,            num_classes=len(classes),
#             EVENT_DICTIONARY=EVENT_DICTIONARY)

        targets_numpy.append(dense_labels)
        detections_numpy.append(dense_predictions)

        closest_numpy = np.zeros(dense_labels.shape)-1
        #Get the closest action index
        closests_numpy.append(get_closest_action_index(dense_labels, closest_numpy))
        # for c in np.arange(dense_labels.shape[-1]):
        #     indexes = np.where(dense_labels[:,c] != 0)[0].tolist()
        #     if len(indexes) == 0 :
        #         continue
        #     indexes.insert(0,-indexes[0])
        #     indexes.append(2*closest_numpy.shape[0])
        #     for i in np.arange(len(indexes)-2)+1:
        #         start = max(0,(indexes[i-1]+indexes[i])//2)
        #         stop = min(closest_numpy.shape[0], (indexes[i]+indexes[i+1])//2)
        #         closest_numpy[start:stop,c] = dense_labels[indexes[i],c]
        # closests_numpy.append(closest_numpy)

    return compute_performances_mAP(metric, targets_numpy, detections_numpy, closests_numpy, INVERSE_EVENT_DICTIONARY)
    # if metric == "loose": deltas=np.arange(12)*5 + 5
    # elif metric == "tight": deltas=np.arange(5)*1 + 1
    # elif metric == "at1": deltas=np.array([1])
    # elif metric == "at2": deltas=np.array([2]) 
    # elif metric == "at3": deltas=np.array([3]) 
    # elif metric == "at4": deltas=np.array([4]) 
    # elif metric == "at5": deltas=np.array([5]) 

    # # Compute the performances
    # a_mAP, a_mAP_per_class = average_mAP(targets_numpy, 
    # detections_numpy, closests_numpy,
    # framerate=2, deltas=deltas)
    
    # results = {
    #     "a_mAP": a_mAP,
    #     "a_mAP_per_class": a_mAP_per_class,
    # }

    # rows = []
    # for i in range(len(results['a_mAP_per_class'])):
    #     label = INVERSE_EVENT_DICTIONARY[i]
    #     rows.append((
    #         label,
    #         '{:0.2f}'.format(results['a_mAP_per_class'][i] * 100),
    #     ))
    # rows.append((
    #     'Average mAP',
    #     '{:0.2f}'.format(results['a_mAP'] * 100),
    # ))

    # logging.info("Best Performance at end of training ")
    # logging.info('Metric: ' +  metric)
    # print(tabulate(rows, headers=['', 'Any']))
    # return results

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
        print(cfg.path)
        GT_data = json.load(f)
    # with open(pred_path) as f :
    #     print(pred_path)
    #     pred_data = json.load(f)

    targets_numpy = list()
    detections_numpy = list()
    closests_numpy = list()

    EVENT_DICTIONARY = {cls: i_cls for i_cls, cls in enumerate(GT_data["labels"])}
    INVERSE_EVENT_DICTIONARY = {i_cls: cls for i_cls, cls in enumerate(GT_data["labels"])}
    for game in tqdm(GT_data["videos"]):

        # fetch labels
        labels = game["annotations"]

        # convert labels to dense vector
        dense_labels = label2vector(labels, 
            num_classes=len(GT_data["labels"]), 
            EVENT_DICTIONARY=EVENT_DICTIONARY)
        
        with open(os.path.join(pred_path,os.path.splitext(game["path_video"])[0],'results_spotting.json')) as f :
            pred_data = json.load(f)
            predictions = pred_data['predictions']
        # for video in pred_data["videos"]:
        #     if video["path_features"] == game["path_features"]:
        #         predictions = video["predictions"]
        #         break

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
    rows = []
    for i in range(len(results['a_mAP_per_class'])):
        label = INVERSE_EVENT_DICTIONARY[i]
        rows.append((
            label,
            '{:0.2f}'.format(results['a_mAP_per_class'][i] * 100),
        ))
    rows.append((
        'Average mAP',
        '{:0.2f}'.format(results['a_mAP'] * 100),
    ))

    logging.info("Best Performance at end of training ")
    logging.info('Metric: ' +  metric)
    print(tabulate(rows, headers=['', 'Any']))
    # logging.info("Best Performance at end of training ")
    # logging.info("a_mAP visibility all: " +  str(a_mAP))
    # logging.info("a_mAP visibility all per class: " +  str( a_mAP_per_class))
    
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