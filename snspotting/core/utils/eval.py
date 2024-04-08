import copy
from collections import defaultdict
import json
import logging
import os
import zipfile
import numpy as np
from tabulate import tabulate
from tqdm import tqdm
from torch.utils.data import DataLoader
from snspotting.core.utils.io import load_json, store_gz_json, store_json
from snspotting.core.utils.score import compute_mAPs


class ErrorStat:

    def __init__(self):
        self._total = 0
        self._err = 0

    def update(self, true, pred):
        self._err += np.sum(true != pred)
        self._total += true.shape[0]

    def get(self):
        return self._err / self._total

    def get_acc(self):
        return 1. - self._get()


class ForegroundF1:

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
        dataset, classes, pred_dict, high_recall_score_threshold=0.01
):
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
        assert np.min(support[1:]) > 0, (video, support[1:].tolist())
        scores[1:] /= support[1:, None]
        pred = np.argmax(scores[1:], axis=1)
        err.update(label[1:], pred)

        pred_scores[video] = scores.tolist()

        events = []
        events_high_recall = []
        # for i in range(1,pred.shape[0]):
        for i in range(pred.shape[0]):
            # f1.update(label[i], pred[i])
            f1.update(label[i+1], pred[i])

            if pred[i] != 0:
                events.append({
                    'label': classes_inv[pred[i]],
                    # 'frame': i,
                    'frame': i + 1,
                    'score': scores[i + 1, pred[i]].item()
                    # 'score': scores[i, pred[i]].item()
                })

            for j in classes_inv:
                if scores[i+1, j] >= high_recall_score_threshold:
                # if scores[i, j] >= high_recall_score_threshold:
                    events_high_recall.append({
                        'label': classes_inv[j],
                        'frame': i + 1,
                        # 'frame': i,
                        'score': scores[i + 1, j].item()
                        # 'score': scores[i, j].item()
                    })

        pred_events.append({
            'video': video, 'events': events,
            'fps': fps_dict[video]})
        pred_events_high_recall.append({
            'video': video, 'events': events_high_recall,
            'fps': fps_dict[video]})

    return err, f1, pred_events, pred_events_high_recall, pred_scores


def non_maximum_supression(pred, window):
    new_pred = []
    for video_pred in pred:
        events_by_label = defaultdict(list)
        for e in video_pred['events']:
            events_by_label[e['label']].append(e)

        events = []
        for v in events_by_label.values():
            for e1 in v:
                for e2 in v:
                    if (
                            e1['frame'] != e2['frame']
                            and abs(e1['frame'] - e2['frame']) <= window
                            and e1['score'] < e2['score']
                    ):
                        # Found another prediction in the window that has a
                        # higher score
                        break
                else:
                    events.append(e1)
        events.sort(key=lambda x: x['frame'])
        new_video_pred = copy.deepcopy(video_pred)
        new_video_pred['events'] = events
        new_video_pred['num_events'] = len(events)
        new_pred.append(new_video_pred)
    return new_pred

def search_best_epoch(work_dir):
    loss = load_json(os.path.join(work_dir,'loss.json'))
    val_mAP = 0
    val = float('inf')
    epoch = -1
    epoch_mAP = -1
    for epoch_loss in loss:
        if epoch_loss["val_mAP"] > val_mAP :
            val_mAP = epoch_loss["val_mAP"]
            epoch_mAP = epoch_loss["epoch"]
        if epoch_loss["val"] < val :
            val = epoch_loss["val"]
            epoch = epoch_loss["epoch"]
    if epoch_mAP != -1: return epoch_mAP
    else: return epoch

def evaluate_e2e(model, dali, dataset, split, classes, save_pred, calc_stats=True,
             save_scores=True):
    pred_dict = {}
    for video, video_len, _ in dataset.videos:
        pred_dict[video] = (
            np.zeros((video_len, len(classes) + 1), np.float32),
            np.zeros(video_len, np.int32))

    # Do not up the batch size if the dataset augments
    batch_size = 1 if dataset.augment else 4

    for clip in tqdm(dataset if dali else DataLoader(
            dataset, num_workers=8, pin_memory=True,
            batch_size=batch_size
    )):
        if batch_size > 1:
            # Batched by dataloader
            _, batch_pred_scores = model.predict(clip['frame'])
            for i in range(clip['frame'].shape[0]):
                video = clip['video'][i]
                scores, support = pred_dict[video]
                pred_scores = batch_pred_scores[i]
                start = clip['start'][i].item()
                if start < 0:
                    pred_scores = pred_scores[-start:, :]
                    start = 0
                end = start + pred_scores.shape[0]
                if end >= scores.shape[0]:
                    end = scores.shape[0]
                    pred_scores = pred_scores[:end - start, :]
                scores[start:end, :] += pred_scores
                support[start:end] += 1

        else:
            # Batched by dataset
            scores, support = pred_dict[clip['video'][0]]

            start = clip['start'][0].item()
            start=start-1
            _, pred_scores = model.predict(clip['frame'][0])
            if start < 0:
                pred_scores = pred_scores[:, -start:, :]
                start = 0
            end = start + pred_scores.shape[1]
            if end >= scores.shape[0]:
                end = scores.shape[0]
                pred_scores = pred_scores[:,:end - start, :]

            print(pred_scores.shape)
            scores[start:end, :] += np.sum(pred_scores, axis=0)
            support[start:end] += pred_scores.shape[0]

    err, f1, pred_events, pred_events_high_recall, pred_scores = \
        process_frame_predictions(dataset, classes, pred_dict)

    avg_mAP = None
    if calc_stats:
        print('=== Results on {} (w/o NMS) ==='.format(split))
        print('Error (frame-level): {:0.2f}\n'.format(err.get() * 100))

        def get_f1_tab_row(str_k):
            k = classes[str_k] if str_k != 'any' else None
            return [str_k, f1.get(k) * 100, *f1.tp_fp_fn(k)]
        rows = [get_f1_tab_row('any')]
        for c in sorted(classes):
            rows.append(get_f1_tab_row(c))
        print(tabulate(rows, headers=['Exact frame', 'F1', 'TP', 'FP', 'FN'],
                    floatfmt='0.2f'))
        print()

        mAPs, _ = compute_mAPs(dataset.labels, pred_events_high_recall)
        avg_mAP = np.mean(mAPs[1:])

    if save_pred is not None:
        store_json(save_pred + '.json', pred_events)
        store_gz_json(save_pred + '.recall.json.gz', pred_events_high_recall)
        if save_scores:
            store_gz_json(save_pred + '.score.json.gz', pred_scores)
    return avg_mAP

def store_eval_files_json(raw_pred, eval_dir):
    video_pred = defaultdict(list)
    for obj in raw_pred:
        video, quality = obj['video'].rsplit('_', 1)
        for event in obj['events']:
            video_pred[video].append({
                'frame': str(event['frame']),
                'label': event['label'],
                'confidence': str(event['score'])
            })
        
    for video, pred in video_pred.items():
        video_out_dir = os.path.join(eval_dir, video)
        os.makedirs(video_out_dir, exist_ok= True)
        store_json(os.path.join(video_out_dir, 'results_spotting.json'), {
            'UrlLocal': video, 'predictions': pred
        }, pretty=True)
        
def label2vector_e2e(labels, vector_size, num_classes=17, framerate=2, version=2, EVENT_DICTIONARY={}):

    # vector_size = 90*60*framerate

    dense_labels = np.zeros((vector_size, num_classes))

    for annotation in labels:

        event = annotation["label"]

        # half = int(time[0])

        frame = int(annotation["frame"])

        label = EVENT_DICTIONARY[event]

        frame = min(frame, vector_size-1)
        dense_labels[frame][label] = 1


    return dense_labels

def predictions2vector_e2e(predictions, vector_size, num_classes=17, version=2, framerate=2, EVENT_DICTIONARY={}):

    # vector_size = 90*60*framerate

    dense_predictions = np.zeros((vector_size, num_classes))-1

    for annotation in predictions:

        event = annotation["label"]

        # half = int(annotation["half"])

        frame = int(annotation["frame"])

        label = EVENT_DICTIONARY[event]

        frame = min(frame, vector_size-1)
        dense_predictions[frame][label] = annotation["confidence"]

    return dense_predictions 
def label2vector(labels, num_classes=17, framerate=2, version=2, EVENT_DICTIONARY={}, vector_size = None):

    vector_size = 90*60*framerate if vector_size is None else vector_size

    dense_labels = np.zeros((vector_size, num_classes))

    for annotation in labels:
        
        event = annotation["label"]
        if "frame" in annotation:
            frame = int(annotation["frame"])
        
        else:
            time = annotation["gameTime"]
            

            # half = int(time[0])

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

def predictions2vector(predictions, num_classes=17, version=2, framerate=2, EVENT_DICTIONARY={} , vector_size = None):

    vector_size = 90*60*framerate if vector_size is None else vector_size

    dense_predictions = np.zeros((vector_size, num_classes))-1

    for annotation in predictions:

        event = annotation["label"]

        if "frame" in annotation:
            frame = int(annotation["frame"])
        else: 
            time = int(annotation["position"])
            
            # half = int(annotation["half"])

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

def get_closest_action_index(dense_labels,closest_numpy):
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
    return closest_numpy

def compute_performances_mAP(metric, targets_numpy, detections_numpy, closests_numpy, INVERSE_EVENT_DICTIONARY):
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
    return results