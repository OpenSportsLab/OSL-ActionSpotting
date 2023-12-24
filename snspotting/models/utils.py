import json
import os
import logging
import zipfile
import numpy as np

from SoccerNet.Evaluation.utils import INVERSE_EVENT_DICTIONARY_V2, INVERSE_EVENT_DICTIONARY_V1

def create_folders(split, work_dir, overwrite):
    # Create folder name and zip file name
    output_folder=f"results_spotting_{'_'.join(split)}"
    output_results=os.path.join(work_dir, f"{output_folder}.zip")
    stop_predict = False
    # Prevent overwriting existing results
    if os.path.exists(output_results) and not overwrite:
        logging.warning("Results already exists in zip format. Use [overwrite=True] to overwrite the previous results.The inference will not run over the previous results.")
        stop_predict=True
        # return output_results
    return output_folder, output_results, stop_predict

def timestamp_half(model,feat_half,BS):
        timestamp_long_half = []
        for b in range(int(np.ceil(len(feat_half)/BS))):
            start_frame = BS*b
            end_frame = BS*(b+1) if BS * \
                (b+1) < len(feat_half) else len(feat_half)
            feat = feat_half[start_frame:end_frame]
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

def zipResults(zip_path, target_dir, filename="results_spotting.json"):            
    zipobj = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)
    rootlen = len(target_dir) + 1
    for base, dirs, files in os.walk(target_dir):
        for file in files:
            if file == filename:
                fn = os.path.join(base, file)
                zipobj.write(fn, fn[rootlen:])
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
