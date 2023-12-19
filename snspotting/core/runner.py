import torch
import os

# from .inference import *



def build_runner(cfg, model=None, default_args=None):
    """Build a runner from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        default_args (dict | None, optional): Default initialization arguments.
            Default: None.

    Returns:
        runner: The constructed runner.
    """
    if cfg.type == "runner_pooling":
        runner = Runner(cfg=cfg,
                        infer_dataset=infer_dataset,
                        infer_game=infer_game,
                        infer_features=infer_features,
                        infer_video=infer_video)
    elif cfg.type == "runner_CALF":
        runner = Runner(cfg=cfg,
                        infer_dataset=infer_dataset_CALF,
                        infer_game=None,
                        infer_features=None,
                        infer_video=None)
    elif cfg.type == "runner_JSON":
        runner = Runner(cfg=cfg,
                        infer_dataset=infer_dataset_JSON,
                        infer_game=infer_game,
                        infer_features=infer_features,
                        infer_video=infer_video) 
    else:
        runner = None
    return runner


class Runner():
    def __init__(self, cfg, 
                infer_dataset, 
                infer_game,
                infer_features,
                infer_video):
        self.infer_dataset = infer_dataset
        self.infer_game = infer_game
        self.infer_features = infer_features
        self.infer_video = infer_video


import logging
import os
import zipfile
import json
import time
from tqdm import tqdm
import torch
import numpy as np

from SoccerNet.Evaluation.utils import AverageMeter, EVENT_DICTIONARY_V2, INVERSE_EVENT_DICTIONARY_V2
from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V1, INVERSE_EVENT_DICTIONARY_V1


def infer_dataset_JSON(cfg, 
                dataloader, 
                model, 
                confidence_threshold=0.0, 
                overwrite=False):

    # Create folder name and zip file name
    output_folder=f"results_spotting_{'_'.join(cfg.dataset.test.split)}"
    output_results=os.path.join(cfg.work_dir, f"{output_folder}.zip")
    output_results_json=os.path.join(cfg.work_dir, f"{output_folder}.json")
    
    # Prevent overwriting existing results
    if os.path.exists(output_results_json) and not overwrite:
        logging.warning("Results already exists in zip format. Use [overwrite=True] to overwrite the previous results.The inference will not run over the previous results.")
        return output_results_json
    
    batch_time = AverageMeter()
    data_time = AverageMeter()

    spotting_predictions = list()

    # put model in eval mode
    model.eval()

    predictions = {}
    predictions["videos"] = []
    predictions["version"] = 1
    predictions["date"] = "2023.08.31"
    predictions["labels"] = dataloader.dataset.classes
    

    end = time.time()
    with tqdm(enumerate(dataloader), total=len(dataloader)) as t:
        for i, (game_ID, features, labels) in t:

            # measure data loading time
            data_time.update(time.time() - end)

            # Batch size of 1
            game_ID = game_ID[0]
            features = features.squeeze(0)

            # Compute the output for batches of frames
            BS = 256
            timestamp_long = []
            for b in range(int(np.ceil(len(features)/BS))):
                start_frame = BS*b
                end_frame = BS*(b+1) if BS * \
                    (b+1) < len(features) else len(features)
                feat = features[start_frame:end_frame].cuda()
                output = model(feat).cpu().detach().numpy()
                timestamp_long.append(output)
            timestamp_long = np.concatenate(timestamp_long)


            timestamp_long = timestamp_long[:, 1:]

            spotting_predictions.append(timestamp_long)

            batch_time.update(time.time() - end)
            end = time.time()

            desc = f'Test (spot.): '
            desc += f'Time {batch_time.avg:.3f}s '
            desc += f'(it:{batch_time.val:.3f}s) '
            desc += f'Data:{data_time.avg:.3f}s '
            desc += f'(it:{data_time.val:.3f}s) '
            t.set_description(desc)



            def get_spot_from_NMS(Input, window=60, thresh=0.0):

                detections_tmp = np.copy(Input)
                indexes = []
                MaxValues = []
                while(np.max(detections_tmp) >= thresh):

                    # Get the max remaining index and value
                    max_value = np.max(detections_tmp)
                    max_index = np.argmax(detections_tmp)
                    MaxValues.append(max_value)
                    indexes.append(max_index)
                    # detections_NMS[max_index,i] = max_value

                    nms_from = int(np.maximum(-(window/2)+max_index,0))
                    nms_to = int(np.minimum(max_index+int(window/2), len(detections_tmp)))
                    detections_tmp[nms_from:nms_to] = -1

                return np.transpose([indexes, MaxValues])

            framerate = dataloader.dataset.framerate
            get_spot = get_spot_from_NMS

            json_data = dict()
            json_data["path_features"] = game_ID
            json_data["predictions"] = list()

            for l in range(dataloader.dataset.num_classes):
                spots = get_spot(
                    timestamp_long[:, l], window=cfg.model.post_proc.NMS_window*cfg.model.backbone.framerate, thresh=cfg.model.post_proc.NMS_threshold)
                for spot in spots:
                    # print("spot", int(spot[0]), spot[1], spot)
                    frame_index = int(spot[0])
                    confidence = spot[1]
                    if confidence < confidence_threshold:
                        continue
                    # confidence = predictions_half_1[frame_index, l]

                    seconds = int((frame_index//framerate)%60)
                    minutes = int((frame_index//framerate)//60)

                    prediction_data = dict()
                    prediction_data["gameTime"] = f"1 - {minutes:02.0f}:{seconds:02.0f}"
                    prediction_data["label"] = dataloader.dataset.classes[l]
                    prediction_data["position"] = str(int((frame_index/framerate)*1000))
                    prediction_data["half"] = str(1)
                    prediction_data["confidence"] = str(confidence)
                    json_data["predictions"].append(prediction_data)
        
            json_data["predictions"] = sorted(json_data["predictions"], key=lambda x: int(x['position']))
            json_data["predictions"] = sorted(json_data["predictions"], key=lambda x: int(x['half']))


            predictions["videos"].append(json_data)


    with open(output_results_json, 'w') as output_file:
        json.dump(predictions, output_file, indent=4)

    return output_results_json


def timestamp_half(feat_half,model,BS):
    timestamp_long_half = []
    for b in range(int(np.ceil(len(feat_half)/BS))):
        start_frame = BS*b
        end_frame = BS*(b+1) if BS * \
            (b+1) < len(feat_half) else len(feat_half)
        feat = feat_half[start_frame:end_frame].cuda()
        output = model(feat).cpu().detach().numpy()
        timestamp_long_half.append(output)
    return np.concatenate(timestamp_long_half)

def get_spot_from_NMS(Input, window=60, thresh=0.0):
    detections_tmp = np.copy(Input)
    indexes = []
    MaxValues = []
    while(np.max(detections_tmp) >= thresh):

        # Get the max remaining index and value
        max_value = np.max(detections_tmp)
        max_index = np.argmax(detections_tmp)
        MaxValues.append(max_value)
        indexes.append(max_index)
        # detections_NMS[max_index,i] = max_value

        nms_from = int(np.maximum(-(window/2)+max_index,0))
        nms_to = int(np.minimum(max_index+int(window/2), len(detections_tmp)))
        detections_tmp[nms_from:nms_to] = -1

    return np.transpose([indexes, MaxValues])

def zipResults(zip_path, target_dir, filename="results_spotting.json"):            
    zipobj = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)
    rootlen = len(target_dir) + 1
    for base, dirs, files in os.walk(target_dir):
        for file in files:
            if file == filename:
                fn = os.path.join(base, file)
                zipobj.write(fn, fn[rootlen:])

def infer_dataset(cfg, dataloader, model, confidence_threshold=0.0, overwrite=False):
    # Create folder name and zip file name
    output_folder=f"results_spotting_{'_'.join(cfg.dataset.test.split)}"
    output_results=os.path.join(cfg.work_dir, f"{output_folder}.zip")

    # Prevent overwriting existing results
    if os.path.exists(output_results) and not overwrite:
        logging.warning("Results already exists in zip format. Use [overwrite=True] to overwrite the previous results.The inference will not run over the previous results.")
        return output_results

    batch_time = AverageMeter()
    data_time = AverageMeter()

    spotting_predictions = list()

    # put model in eval mode
    model.eval()

    end = time.time()

    with tqdm(enumerate(dataloader), total=len(dataloader)) as t:
        for i, (game_ID, feat_half1, feat_half2, label_half1, label_half2) in t:
            # measure data loading time
            data_time.update(time.time() - end)

            game_ID = game_ID[0]
            feat_half1 = feat_half1.squeeze(0)
            feat_half2 = feat_half2.squeeze(0)

            # Compute the output for batches of frames
            BS = 256
            timestamp_long_half_1 = timestamp_half(feat_half1,model,BS)
            timestamp_long_half_2 = timestamp_half(feat_half2,model,BS)
            
            timestamp_long_half_1 = timestamp_long_half_1[:, 1:]
            timestamp_long_half_2 = timestamp_long_half_2[:, 1:]

            spotting_predictions.append(timestamp_long_half_1)
            spotting_predictions.append(timestamp_long_half_2)

            batch_time.update(time.time() - end)
            end = time.time()

            desc = f'Test (spot.): '
            desc += f'Time {batch_time.avg:.3f}s '
            desc += f'(it:{batch_time.val:.3f}s) '
            desc += f'Data:{data_time.avg:.3f}s '
            desc += f'(it:{data_time.val:.3f}s) '
            t.set_description(desc)

            framerate = dataloader.dataset.framerate
            get_spot = get_spot_from_NMS

            json_data = get_json_data(False,game_ID=game_ID)

            for half, timestamp in enumerate([timestamp_long_half_1, timestamp_long_half_2]):
                for l in range(dataloader.dataset.num_classes):
                    spots = get_spot(
                        timestamp[:, l], window=cfg.model.post_proc.NMS_window*cfg.model.backbone.framerate, thresh=cfg.model.post_proc.NMS_threshold)
                    for spot in spots:
                        # print("spot", int(spot[0]), spot[1], spot)
                        frame_index = int(spot[0])
                        confidence = spot[1]
                        if confidence < confidence_threshold:
                            continue
                        
                        json_data["predictions"].append(get_prediction_data(False,frame_index,framerate,half=half,version=dataloader.dataset.version,l=l,confidence=confidence))
            
                json_data["predictions"] = sorted(json_data["predictions"], key=lambda x: int(x['position']))
                json_data["predictions"] = sorted(json_data["predictions"], key=lambda x: int(x['half']))

            os.makedirs(os.path.join(cfg.work_dir, output_folder, game_ID), exist_ok=True)
            with open(os.path.join(cfg.work_dir, output_folder, game_ID, "results_spotting.json"), 'w') as output_file:
                json.dump(json_data, output_file, indent=4)
    # zip folder
    zipResults(zip_path = output_results,
            target_dir = os.path.join(cfg.work_dir, output_folder),
            filename="results_spotting.json")

    return output_results

def infer_dataset_CALF(cfg, dataloader, model, confidence_threshold=0.0, overwrite=False):
    # Create folder name and zip file name
    output_folder=f"results_spotting_{'_'.join(cfg.dataset.test.split)}"
    output_results=os.path.join(cfg.work_dir, f"{output_folder}.zip")
    
    # Prevent overwriting existing results
    if os.path.exists(output_results) and not overwrite:
        logging.warning("Results already exists in zip format. Use [overwrite=True] to overwrite the previous results.The inference will not run over the previous results.")
        return output_results

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    spotting_predictions = list()
    spotting_grountruth = list()
    spotting_grountruth_visibility = list()
    segmentation_predictions = list()
    chunk_size = model.chunk_size
    receptive_field = model.receptive_field

    model.eval()

    end = time.time()
    with tqdm(enumerate(dataloader), total=len(dataloader)) as t:
        for i,(feat_half1, feat_half2, label_half1, label_half2) in t:
            # measure data loading time
            data_time.update(time.time() - end)

            feat_half1 = feat_half1.cuda()
            label_half1 = label_half1.float().squeeze(0)
            feat_half2 = feat_half2.cuda()
            label_half2 = label_half2.float().squeeze(0)

            feat_half1 = feat_half1.squeeze(0)
            feat_half2 = feat_half2.squeeze(0)

            feat_half1=feat_half1.unsqueeze(1)
            feat_half2=feat_half2.unsqueeze(1)

            # Compute the output
            output_segmentation_half_1, output_spotting_half_1 = model(feat_half1)
            output_segmentation_half_2, output_spotting_half_2 = model(feat_half2)

            timestamp_long_half_1 = timestamps2long(output_spotting_half_1.cpu().detach(), label_half1.size()[0], chunk_size, receptive_field)
            timestamp_long_half_2 = timestamps2long(output_spotting_half_2.cpu().detach(), label_half2.size()[0], chunk_size, receptive_field)
            segmentation_long_half_1 = batch2long(output_segmentation_half_1.cpu().detach(), label_half1.size()[0], chunk_size, receptive_field)
            segmentation_long_half_2 = batch2long(output_segmentation_half_2.cpu().detach(), label_half2.size()[0], chunk_size, receptive_field)

            spotting_grountruth.append(torch.abs(label_half1))
            spotting_grountruth.append(torch.abs(label_half2))
            spotting_grountruth_visibility.append(label_half1)
            spotting_grountruth_visibility.append(label_half2)
            spotting_predictions.append(timestamp_long_half_1)
            spotting_predictions.append(timestamp_long_half_2)
            segmentation_predictions.append(segmentation_long_half_1)
            segmentation_predictions.append(segmentation_long_half_2)

    # Transformation to numpy for evaluation
    targets_numpy = list()
    closests_numpy = list()
    detections_numpy = list()
    for target, detection in zip(spotting_grountruth_visibility,spotting_predictions):
        target_numpy = target.numpy()
        targets_numpy.append(target_numpy)
        detections_numpy.append(NMS(detection.numpy(), 20*model.framerate))
        closest_numpy = np.zeros(target_numpy.shape)-1
        #Get the closest action index
        for c in np.arange(target_numpy.shape[-1]):
            indexes = np.where(target_numpy[:,c] != 0)[0].tolist()
            if len(indexes) == 0 :
                continue
            indexes.insert(0,-indexes[0])
            indexes.append(2*closest_numpy.shape[0])
            for i in np.arange(len(indexes)-2)+1:
                start = max(0,(indexes[i-1]+indexes[i])//2)
                stop = min(closest_numpy.shape[0], (indexes[i]+indexes[i+1])//2)
                closest_numpy[start:stop,c] = target_numpy[indexes[i],c]
        closests_numpy.append(closest_numpy)

    # Save the predictions to the json format
    # if save_predictions:
    list_game = getListGames(dataloader.dataset.split)
    for index in np.arange(len(list_game)):
        predictions2json(detections_numpy[index*2], detections_numpy[(index*2)+1],cfg.work_dir+"/"+output_folder+"/", list_game[index], model.framerate)

    # zip folder
    zipResults(zip_path = output_results,
            target_dir = os.path.join(cfg.work_dir, output_folder),
            filename="results_spotting.json")

    return output_results

def infer_dataset(cfg, dataloader, model, confidence_threshold=0.0, overwrite=False, calf = False):

    # Create folder name and zip file name
    output_folder=f"results_spotting_{'_'.join(cfg.dataset.test.split)}"
    output_results=os.path.join(cfg.work_dir, f"{output_folder}.zip")
    
    # Prevent overwriting existing results
    if os.path.exists(output_results) and not overwrite:
        logging.warning("Results already exists in zip format. Use [overwrite=True] to overwrite the previous results.The inference will not run over the previous results.")
        return output_results

    batch_time = AverageMeter()
    data_time = AverageMeter()
    if calf : losses = AverageMeter()

    spotting_predictions = list()
    if calf : 
        spotting_grountruth = list()
        spotting_grountruth_visibility = list()
        segmentation_predictions = list()
        chunk_size = model.chunk_size
        receptive_field = model.receptive_field
    

    # put model in eval mode
    model.eval()


    end = time.time()
    #infer_data set loop
    with tqdm(enumerate(dataloader), total=len(dataloader)) as t:
        for i, tuples in t:
            if calf :
                feat_half1, feat_half2, label_half1, label_half2 = tuples
            else :
                game_ID, feat_half1, feat_half2, label_half1, label_half2 = tuples
            # measure data loading time
            data_time.update(time.time() - end)

            if calf:
                feat_half1 = feat_half1.cuda()
                label_half1 = label_half1.float().squeeze(0)
                feat_half2 = feat_half2.cuda()
                label_half2 = label_half2.float().squeeze(0)

            
            # Batch size of 1
            if not calf : game_ID = game_ID[0]
            feat_half1 = feat_half1.squeeze(0)
            feat_half2 = feat_half2.squeeze(0)

            if calf :
                feat_half1=feat_half1.unsqueeze(1)
                feat_half2=feat_half2.unsqueeze(1)
            
            if calf:
                # Compute the output
                output_segmentation_half_1, output_spotting_half_1 = model(feat_half1)
                output_segmentation_half_2, output_spotting_half_2 = model(feat_half2)

                timestamp_long_half_1 = timestamps2long(output_spotting_half_1.cpu().detach(), label_half1.size()[0], chunk_size, receptive_field)
                timestamp_long_half_2 = timestamps2long(output_spotting_half_2.cpu().detach(), label_half2.size()[0], chunk_size, receptive_field)
                segmentation_long_half_1 = batch2long(output_segmentation_half_1.cpu().detach(), label_half1.size()[0], chunk_size, receptive_field)
                segmentation_long_half_2 = batch2long(output_segmentation_half_2.cpu().detach(), label_half2.size()[0], chunk_size, receptive_field)

                spotting_grountruth.append(torch.abs(label_half1))
                spotting_grountruth.append(torch.abs(label_half2))
                spotting_grountruth_visibility.append(label_half1)
                spotting_grountruth_visibility.append(label_half2)
                spotting_predictions.append(timestamp_long_half_1)
                spotting_predictions.append(timestamp_long_half_2)
                segmentation_predictions.append(segmentation_long_half_1)
                segmentation_predictions.append(segmentation_long_half_2)
            
            else:
                # Compute the output for batches of frames
                BS = 256
                timestamp_long_half_1 = timestamp_half(feat_half1,model,BS)
                timestamp_long_half_2 = timestamp_half(feat_half2,model,BS)
                
                timestamp_long_half_1 = timestamp_long_half_1[:, 1:]
                timestamp_long_half_2 = timestamp_long_half_2[:, 1:]

                spotting_predictions.append(timestamp_long_half_1)
                spotting_predictions.append(timestamp_long_half_2)

                batch_time.update(time.time() - end)
                end = time.time()

                desc = f'Test (spot.): '
                desc += f'Time {batch_time.avg:.3f}s '
                desc += f'(it:{batch_time.val:.3f}s) '
                desc += f'Data:{data_time.avg:.3f}s '
                desc += f'(it:{data_time.val:.3f}s) '
                t.set_description(desc)

                framerate = dataloader.dataset.framerate
                get_spot = get_spot_from_NMS

                json_data = get_json_data(False,game_ID=game_ID)

                for half, timestamp in enumerate([timestamp_long_half_1, timestamp_long_half_2]):
                    for l in range(dataloader.dataset.num_classes):
                        spots = get_spot(
                            timestamp[:, l], window=cfg.model.post_proc.NMS_window*cfg.model.backbone.framerate, thresh=cfg.model.post_proc.NMS_threshold)
                        for spot in spots:
                            # print("spot", int(spot[0]), spot[1], spot)
                            frame_index = int(spot[0])
                            confidence = spot[1]
                            if confidence < confidence_threshold:
                                continue
                            
                            json_data["predictions"].append(get_prediction_data(False,frame_index,framerate,half=half,version=dataloader.dataset.version,l=l,confidence=confidence))
                
                    json_data["predictions"] = sorted(json_data["predictions"], key=lambda x: int(x['position']))
                    json_data["predictions"] = sorted(json_data["predictions"], key=lambda x: int(x['half']))

                os.makedirs(os.path.join(cfg.work_dir, output_folder, game_ID), exist_ok=True)
                with open(os.path.join(cfg.work_dir, output_folder, game_ID, "results_spotting.json"), 'w') as output_file:
                    json.dump(json_data, output_file, indent=4)
    if calf :
        # Transformation to numpy for evaluation
        targets_numpy = list()
        closests_numpy = list()
        detections_numpy = list()
        for target, detection in zip(spotting_grountruth_visibility,spotting_predictions):
            target_numpy = target.numpy()
            targets_numpy.append(target_numpy)
            detections_numpy.append(NMS(detection.numpy(), 20*model.framerate))
            closest_numpy = np.zeros(target_numpy.shape)-1
            #Get the closest action index
            for c in np.arange(target_numpy.shape[-1]):
                indexes = np.where(target_numpy[:,c] != 0)[0].tolist()
                if len(indexes) == 0 :
                    continue
                indexes.insert(0,-indexes[0])
                indexes.append(2*closest_numpy.shape[0])
                for i in np.arange(len(indexes)-2)+1:
                    start = max(0,(indexes[i-1]+indexes[i])//2)
                    stop = min(closest_numpy.shape[0], (indexes[i]+indexes[i+1])//2)
                    closest_numpy[start:stop,c] = target_numpy[indexes[i],c]
            closests_numpy.append(closest_numpy)

        # Save the predictions to the json format
        # if save_predictions:
        list_game = getListGames(dataloader.dataset.split)
        for index in np.arange(len(list_game)):
            predictions2json(detections_numpy[index*2], detections_numpy[(index*2)+1],cfg.work_dir+"/"+output_folder+"/", list_game[index], model.framerate)

    # zip folder
    zipResults(zip_path = output_results,
            target_dir = os.path.join(cfg.work_dir, output_folder),
            filename="results_spotting.json")

    return output_results


from snspotting.datasets.soccernet import feats2clip

def infer_game(cfg, game_folder, model, confidence_threshold=0.5, overwrite=False):

    feature_half1 = os.path.join(game_folder, "1_"+cfg.dataset.test.features)
    feature_half2 = os.path.join(game_folder, "2_"+cfg.dataset.test.features)

    results_half1 = infer_features(cfg, feature_half1, model, confidence_threshold, overwrite)
    results_half2 = infer_features(cfg, feature_half2, model, confidence_threshold, overwrite)

    json_data = dict()
    json_data["UrlLocal"] = game_folder
    json_data["predictions"] = list()

    for pred in results_half1["predictions"]:
        pred["half"] = "1"
        pred["gameTime"] = "1" + pred["gameTime"][1:]
        json_data["predictions"].append(pred)

    for pred in results_half2["predictions"]:
        pred["half"] = "2"
        pred["gameTime"] = "2" + pred["gameTime"][1:]
        json_data["predictions"].append(pred)

    return json_data


def infer_features(cfg, features_file, model, confidence_threshold=0.5, overwrite=False):
      
    features = np.load(features_file)
    features = feats2clip(torch.from_numpy(features), 
                    stride=1, off=int(cfg.dataset.test.window_size/2), 
                    clip_length=cfg.dataset.test.window_size)
    
    # Compute the output for batches of frames
    spotting_predictions = list()
    BS = 256
    timestamp_long = []
    for b in range(int(np.ceil(len(features)/BS))):
        start_frame = BS*b
        end_frame = BS*(b+1) if BS * \
            (b+1) < len(features) else len(features)
        feat = features[start_frame:end_frame].cuda()
        output = model(feat).cpu().detach().numpy()
        timestamp_long.append(output)
    timestamp_long = np.concatenate(timestamp_long)

    timestamp_long = timestamp_long[:, 1:]

    spotting_predictions.append(timestamp_long)




    def get_spot_from_NMS(Input, window=60, thresh=0.0):

        detections_tmp = np.copy(Input)
        indexes = []
        MaxValues = []
        while(np.max(detections_tmp) >= thresh):

            # Get the max remaining index and value
            max_value = np.max(detections_tmp)
            max_index = np.argmax(detections_tmp)
            MaxValues.append(max_value)
            indexes.append(max_index)
            # detections_NMS[max_index,i] = max_value

            nms_from = int(np.maximum(-(window/2)+max_index,0))
            nms_to = int(np.minimum(max_index+int(window/2), len(detections_tmp)))
            detections_tmp[nms_from:nms_to] = -1

        return np.transpose([indexes, MaxValues])

    # framerate = dataloader.dataset.framerate
    framerate = cfg.dataset.test.framerate
    get_spot = get_spot_from_NMS

    json_data = dict()
    json_data["UrlLocal"] = features_file
    json_data["predictions"] = list()

    for half, timestamp in enumerate([timestamp_long]):
        # for l in range(dataloader.dataset.num_classes):
        for l in range(len(cfg.dataset.test.classes)):
            spots = get_spot(
                timestamp[:, l], window=cfg.model.post_proc.NMS_window*cfg.model.backbone.framerate, thresh=cfg.model.post_proc.NMS_threshold)
            for spot in spots:
                # print("spot", int(spot[0]), spot[1], spot)
                frame_index = int(spot[0])
                confidence = spot[1]
                if confidence < confidence_threshold:
                    continue
                # confidence = predictions_half_1[frame_index, l]

                seconds = int((frame_index//framerate)%60)
                minutes = int((frame_index//framerate)//60)

                prediction_data = dict()
                prediction_data["gameTime"] = f"{half+1} - {minutes:02.0f}:{seconds:02.0f}"
                if cfg.dataset.test.version == 2:
                    prediction_data["label"] = INVERSE_EVENT_DICTIONARY_V2[l]
                else:
                    prediction_data["label"] = INVERSE_EVENT_DICTIONARY_V1[l]
                prediction_data["position"] = str(int((frame_index/framerate)*1000))
                prediction_data["half"] = str(half+1)
                prediction_data["confidence"] = str(confidence)
                json_data["predictions"].append(prediction_data)
    
    json_data["predictions"] = sorted(json_data["predictions"], key=lambda x: int(x['position']))

    return json_data


def infer_video(cfg, video, model, confidence_threshold=0.5, overwrite=False):
    return

## CODE CALF
from snspotting.datasets.soccernet import timestamps2long, batch2long
from SoccerNet.Downloader import getListGames

def NMS(detections, delta):
    
    # Array to put the results of the NMS
    detections_tmp = np.copy(detections)
    detections_NMS = np.zeros(detections.shape)-1

    # Loop over all classes
    for i in np.arange(detections.shape[-1]):
        # Stopping condition
        while(np.max(detections_tmp[:,i]) >= 0):

            # Get the max remaining index and value
            max_value = np.max(detections_tmp[:,i])
            max_index = np.argmax(detections_tmp[:,i])

            detections_NMS[max_index,i] = max_value

            detections_tmp[int(np.maximum(-(delta/2)+max_index,0)): int(np.minimum(max_index+int(delta/2), detections.shape[0])) ,i] = -1

    return detections_NMS

def get_json_data(calf,game_info=None,game_ID=None):
    json_data = dict()
    json_data["UrlLocal"] = game_info if calf else game_ID
    json_data["predictions"] = list()
    return json_data

def get_prediction_data(calf,frame_index, framerate, class_index=None, confidence=None, half=None, l=None, version=None, half_1=None):
    seconds = int((frame_index//framerate)%60)
    minutes = int((frame_index//framerate)//60)

    prediction_data = dict()
    prediction_data["gameTime"] = (str(1 if half_1 else 2 ) + " - " + str(minutes) + ":" + str(seconds)) if calf else f"{half+1} - {minutes:02.0f}:{seconds:02.0f}"
    prediction_data["label"] = INVERSE_EVENT_DICTIONARY_V2[class_index if calf else l] if version == 2 else INVERSE_EVENT_DICTIONARY_V1[l]
    prediction_data["position"] = str(int((frame_index/framerate)*1000))
    prediction_data["half"] = str(1 if half_1 else 2) if calf else str(half+1)
    prediction_data["confidence"] = str(confidence)

    return prediction_data
    
def predictions2json(predictions_half_1, predictions_half_2, output_path, game_info, framerate=2):

    os.makedirs(output_path + game_info, exist_ok=True)
    output_file_path = output_path + game_info + "/results_spotting.json"

    frames_half_1, class_half_1 = np.where(predictions_half_1 >= 0)
    frames_half_2, class_half_2 = np.where(predictions_half_2 >= 0)
    
    json_data = get_json_data(True,game_info=game_info)
    
    for frame_index, class_index in zip(frames_half_1, class_half_1):

        confidence = predictions_half_1[frame_index, class_index]

        json_data["predictions"].append(get_prediction_data(True,frame_index,framerate,class_index=class_index,confidence=confidence,version=2,half_1=True))

    for frame_index, class_index in zip(frames_half_2, class_half_2):

        confidence = predictions_half_2[frame_index, class_index]

        json_data["predictions"].append(get_prediction_data(True,frame_index,framerate,class_index=class_index,confidence=confidence,version=2,half_1=False))
    
    with open(output_file_path, 'w') as output_file:
        json.dump(json_data, output_file, indent=4)

    # # Compute the performances
    # a_mAP, a_mAP_per_class, a_mAP_visible, a_mAP_per_class_visible, a_mAP_unshown, a_mAP_per_class_unshown = average_mAP(targets_numpy, detections_numpy, closests_numpy, model.framerate)
    
    # print("Average mAP: ", a_mAP)
    # print("Average mAP visible: ", a_mAP_visible)
    # print("Average mAP unshown: ", a_mAP_unshown)
    # print("Average mAP per class: ", a_mAP_per_class)
    # print("Average mAP visible per class: ", a_mAP_per_class_visible)
    # print("Average mAP unshown per class: ", a_mAP_per_class_unshown)

    # return a_mAP, a_mAP_per_class, a_mAP_visible, a_mAP_per_class_visible, a_mAP_unshown, a_mAP_per_class_unshown

