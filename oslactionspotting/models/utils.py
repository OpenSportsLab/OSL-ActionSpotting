import json
import os
import logging
import zipfile
import numpy as np

from SoccerNet.Evaluation.utils import (
    INVERSE_EVENT_DICTIONARY_V2,
    INVERSE_EVENT_DICTIONARY_V1,
)


def check_if_should_predict(folder_name, work_dir, overwrite):
    """Check if zip file with specified name already exists, if it exists and overwrite is false, it should not predict
    Args:
        folder_name (string): Name of the folder and of the file zip.
        work_dir (string): folder where the zip is located
        overwrite (bool).
    """
    # Create folder name and zip file name
    output_folder = folder_name
    # output_folder=f"results_spotting_{'_'.join(split)}"
    output_results = os.path.join(work_dir, f"{output_folder}.zip")
    stop_predict = False
    # Prevent overwriting existing results
    if os.path.exists(output_results) and not overwrite:
        logging.warning(
            "Results already exists in zip format. Use [overwrite=True] to overwrite the previous results.The inference will not run over the previous results."
        )
        stop_predict = True
        # return output_results
    return output_folder, output_results, stop_predict


def timestamp(model, feat, BS):
    """Compute the timestamps for features using a model and a batch size."""
    timestamp_long = []
    for b in range(int(np.ceil(len(feat) / BS))):
        start_frame = BS * b
        end_frame = BS * (b + 1) if BS * (b + 1) < len(feat) else len(feat)
        feat_tmp = feat[start_frame:end_frame].cuda()
        output = model(feat_tmp).cpu().detach().numpy()
        timestamp_long.append(output)
    return np.concatenate(timestamp_long)


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


def get_json_data(info):
    """Create a dict that is the content of a json file.
    The dict contains the keys:
        -Url which is the name of the feature/video file/game.
        -predictions that will contain a list of predictions.
    """
    json_data = dict()
    json_data["Url"] = info
    json_data["predictions"] = list()
    return json_data


def get_prediction_data(
    calf,
    frame_index,
    framerate,
    class_index=None,
    confidence=None,
    half=None,
    l=None,
    version=None,
    half_1=None,
    runner="runner_JSON",
    inverse_event_dictionary=None,
):
    """Create and return a dict that represents data for an event containing the time at which the event occurs, the name of the event, the position (temporal) and the confidence.
    If data comes from SoccerNet data set modules, which half is also included.
    Args:
        calf (bool): Whether it is for the calf method.
        frame_index (int).
        framerate (int).
        class_index (int): Index of the class with which we will retrieve the name of the class.
            Default: None.
        confidence (float): The confidence for the prediction.
            Default: None.
        half (int): The half.
            Default: None.
        l (int): Index of the class with which we will retrieve the name of the class. Used if it is non calf method.
        version (int): The version of data for the SoccerNet datasets if used.
            Default: None.
        half_1
        runner (string): Which runner is used. "runner_JSON" if data comes from json, "runner_pooling" or ""runner_CALF" if data comes from SoccerNet datasets modules.
        The difference between the first one and the others is that for the first one, we do not include the notion of half and the dict of classes is given.
            Default: "runner_JSON".
        inverse_event_dictionary (dict): Mapping between indexes and classes names. Needed if runner_JSON
            Default: None.
    """
    seconds = int((frame_index // framerate) % 60)
    minutes = int((frame_index // framerate) // 60)
    # print(frame_index,framerate)
    prediction_data = dict()
    if runner == "runner_JSON":
        prediction_data["gameTime"] = f"{minutes:02.0f}:{seconds:02.0f}"
        # print(f"{minutes:02.0f}:{seconds:02.0f}")
    else:
        prediction_data["half"] = str(1 if half_1 else 2) if calf else str(half + 1)
        prediction_data["gameTime"] = (
            (str(1 if half_1 else 2) + " - " + str(minutes) + ":" + str(seconds))
            if calf
            else f"{half+1} - {minutes:02.0f}:{seconds:02.0f}"
        )
    if runner == "runner_JSON":
        prediction_data["label"] = inverse_event_dictionary[class_index if calf else l]
    else:
        prediction_data["label"] = (
            INVERSE_EVENT_DICTIONARY_V2[class_index if calf else l]
            if version == 2
            else INVERSE_EVENT_DICTIONARY_V1[l]
        )
    prediction_data["position"] = int((frame_index / framerate) * 1000)
    prediction_data["confidence"] = confidence

    return prediction_data


def zipResults(zip_path, target_dir, filename="results_spotting.json"):
    """Zip a folder of predictions into a zip file."""
    zipobj = zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED)
    rootlen = len(target_dir) + 1
    for base, dirs, files in os.walk(target_dir):
        for file in files:
            if file == filename:
                fn = os.path.join(base, file)
                zipobj.write(fn, fn[rootlen:])


def NMS(detections, delta):

    # Array to put the results of the NMS
    detections_tmp = np.copy(detections)
    detections_NMS = np.zeros(detections.shape) - 1

    # Loop over all classes
    for i in np.arange(detections.shape[-1]):
        # Stopping condition
        while np.max(detections_tmp[:, i]) >= 0:

            # Get the max remaining index and value
            max_value = np.max(detections_tmp[:, i])
            max_index = np.argmax(detections_tmp[:, i])

            detections_NMS[max_index, i] = max_value

            detections_tmp[
                int(np.maximum(-(delta / 2) + max_index, 0)) : int(
                    np.minimum(max_index + int(delta / 2), detections.shape[0])
                ),
                i,
            ] = -1

    return detections_NMS


def predictions2json(
    predictions_half_1, predictions_half_2, json_data, output_path, framerate=2
):
    """Construct a list of dict with informations for each prediction, add the list to a json object and saves the json file. Used for runner_CALF.

    Args:
        predictions_half_1: Contains an array of predictions for the first half.
        predictions_half_2: Contains an array of predictions for the first half.
        json_data: The pseudo json object in which we will add predictions.
        output_path: The path of the json file.
        framerate (int).
            Default: 2.
    """
    frames_half_1, class_half_1 = np.where(predictions_half_1 >= 0)
    frames_half_2, class_half_2 = np.where(predictions_half_2 >= 0)

    # json_data = get_json_data(game_info)

    for frame_index, class_index in zip(frames_half_1, class_half_1):

        confidence = predictions_half_1[frame_index, class_index]

        json_data["predictions"].append(
            get_prediction_data(
                True,
                frame_index,
                framerate,
                class_index=class_index,
                confidence=confidence,
                version=2,
                half_1=True,
                runner="runner_CALF",
            )
        )

    for frame_index, class_index in zip(frames_half_2, class_half_2):

        confidence = predictions_half_2[frame_index, class_index]

        json_data["predictions"].append(
            get_prediction_data(
                True,
                frame_index,
                framerate,
                class_index=class_index,
                confidence=confidence,
                version=2,
                half_1=False,
                runner="runner_CALF",
            )
        )

    with open(output_path, "w") as output_file:
        json.dump(json_data, output_file, indent=4)
    return json_data


def predictions2json_runnerjson(
    predictions_video,
    json_data,
    output_path,
    framerate=2,
    inverse_event_dictionary=None,
):
    """Construct a list of dict with informations for each prediction, add the list to a json object and saves the json file. Used for runner_JSON.

    Args:
        predictions_video: Contains an array of predictions for the first half.
        json_data: The pseudo json object in which we will add predictions.
        output_path: The path of the json file.
        framerate (int).
            Default: 2.
        inverse_event_dictionary (dict): Mapping between indexes and classes name.
    """
    frames_video, class_video = np.where(predictions_video >= 0)

    for frame_index, class_index in zip(frames_video, class_video):

        confidence = predictions_video[frame_index, class_index]

        json_data["predictions"].append(
            get_prediction_data(
                True,
                frame_index,
                framerate,
                class_index=class_index,
                confidence=confidence,
                version=2,
                half_1=True,
                runner="runner_JSON",
                inverse_event_dictionary=inverse_event_dictionary,
            )
        )

    with open(output_path, "w") as output_file:
        json.dump(json_data, output_file, indent=4)
    return json_data
