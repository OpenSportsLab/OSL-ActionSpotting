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

            # Batch size of 1
            game_ID = game_ID[0]
            feat_half1 = feat_half1.squeeze(0)
            feat_half2 = feat_half2.squeeze(0)

            # Compute the output for batches of frames
            BS = 256
            timestamp_long_half_1 = []
            for b in range(int(np.ceil(len(feat_half1)/BS))):
                start_frame = BS*b
                end_frame = BS*(b+1) if BS * \
                    (b+1) < len(feat_half1) else len(feat_half1)
                feat = feat_half1[start_frame:end_frame].cuda()
                output = model(feat).cpu().detach().numpy()
                timestamp_long_half_1.append(output)
            timestamp_long_half_1 = np.concatenate(timestamp_long_half_1)

            timestamp_long_half_2 = []
            for b in range(int(np.ceil(len(feat_half2)/BS))):
                start_frame = BS*b
                end_frame = BS*(b+1) if BS * \
                    (b+1) < len(feat_half2) else len(feat_half2)
                feat = feat_half2[start_frame:end_frame].cuda()
                output = model(feat).cpu().detach().numpy()
                timestamp_long_half_2.append(output)
            timestamp_long_half_2 = np.concatenate(timestamp_long_half_2)


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
            json_data["UrlLocal"] = game_ID
            json_data["predictions"] = list()

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
                        # confidence = predictions_half_1[frame_index, l]

                        seconds = int((frame_index//framerate)%60)
                        minutes = int((frame_index//framerate)//60)

                        prediction_data = dict()
                        prediction_data["gameTime"] = f"{half+1} - {minutes:02.0f}:{seconds:02.0f}"
                        if dataloader.dataset.version == 2:
                            prediction_data["label"] = INVERSE_EVENT_DICTIONARY_V2[l]
                        else:
                            prediction_data["label"] = INVERSE_EVENT_DICTIONARY_V1[l]
                        prediction_data["position"] = str(int((frame_index/framerate)*1000))
                        prediction_data["half"] = str(half+1)
                        prediction_data["confidence"] = str(confidence)
                        json_data["predictions"].append(prediction_data)
            
                json_data["predictions"] = sorted(json_data["predictions"], key=lambda x: int(x['position']))
                json_data["predictions"] = sorted(json_data["predictions"], key=lambda x: int(x['half']))

            os.makedirs(os.path.join(cfg.work_dir, output_folder, game_ID), exist_ok=True)
            with open(os.path.join(cfg.work_dir, output_folder, game_ID, "results_spotting.json"), 'w') as output_file:
                json.dump(json_data, output_file, indent=4)


    def zipResults(zip_path, target_dir, filename="results_spotting.json"):            
        zipobj = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)
        rootlen = len(target_dir) + 1
        for base, dirs, files in os.walk(target_dir):
            for file in files:
                if file == filename:
                    fn = os.path.join(base, file)
                    zipobj.write(fn, fn[rootlen:])

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
