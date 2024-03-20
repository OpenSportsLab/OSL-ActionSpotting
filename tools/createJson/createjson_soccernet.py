import json
import os
import glob


# DIR = "/media/giancos/NewVolume/OSL-ActionSpotting/SoccerNet"
SOCCERNET_ORIG_FOLDER = "/media/giancos/NewVolume/OSL-ActionSpotting/SoccerNet"
SOCCERNET_ZIP_FOLDER = "/media/giancos/NewVolume/OSL-ActionSpotting/SoccerNetZip/Json_videos"
# SOCCERNET_ZIP_FOLDER = "/media/giancos/NewVolume/OSL-ActionSpotting/SoccerNetZip/Json_features"

# for split in ["train", "valid", "test", ["train", "valid", "test"]]:

def createjson(split, json_file):
    data = {}
    data["version"]=1
    data["date"]="2023.08.31"
    data["videos"] = []

    # DIR="path/to/SoccerNet/"
    from SoccerNet.utils import getListGames
    for game in getListGames(split):
        for half in [1,2]:
        # for feature in sorted(glob.glob(f'{DIR}/{game}/*_ResNET_TF2_PCA512.npy')):
            feature = os.path.join(SOCCERNET_ORIG_FOLDER, game, f'{half}_ResNET_TF2_PCA512.npy')
            # print(feature)
            label_file = os.path.join(SOCCERNET_ORIG_FOLDER, game, "Labels-v2.json")
            # print(os.path.dirname(feature), "&&&&", os.path.basename(feature))
            with open(label_file) as f :
                label = json.load(f) 
                # print(label)
            
            video = {}    
            # video_file = glob.glob(f'path/to/SoccerNet/*{fold}*.mp4')[0]
            # video_file = os.path.basename(video_file)
            if "1_" in os.path.basename(feature):
                half = 1
            elif "2_" in os.path.basename(feature):
                half = 2

            video["path_video"] = os.path.join(game.replace(" ","_"), f"{half}_224p.mkv")
            video["path_features"] = os.path.join(game.replace(" ","_"), os.path.basename(feature)) # feature #fold+".npy"

            video["annotations"] = [ann for ann in label["annotations"] if int(ann["gameTime"][0]) == int(half)]
            for anno in video["annotations"]:
                anno["position"] = int(anno["position"])
            data["videos"].append(video)

    # labels = [anno["label"] for v in data["videos"] for anno in v["annotations"] ]
    # data["labels"]=list(set(labels))
    data["labels"] = ["Penalty", "Kick-off", "Goal", "Substitution", "Offside", 
    "Shots on target", "Shots off target", "Clearance", "Ball out of play", 
    "Throw-in", "Foul", "Indirect free-kick", "Direct free-kick", "Corner", 
    "Yellow card","Red card", "Yellow->red card",]
    JSON_all = f'{SOCCERNET_ZIP_FOLDER}/{json_file}'
    with open(JSON_all, 'w') as f:
        json.dump(data, f, indent=4)
    # import zipfile
    # zip = f"{SOCCERNET_ZIP_FOLDER}/Json_videos/{split}.zip"
    # file = f"{SOCCERNET_ZIP_FOLDER}/{split}.json"
    # with zipfile.ZipFile(zip, 'a') as zipf:
    #     zipf.write(file, arcname=f"{split}.json")

    

createjson("train", "train.json")
createjson("valid", "valid.json")
createjson("test", "test.json")
# # createjson("challenge", "challenge.json")
# createjson(["train","valid","test"], "all.json")

# JSON_train = 'path/to/SoccerNet/train.json'
# JSON_val = 'path/to/SoccerNet/val.json'
# JSON_test = 'path/to/SoccerNet/test.json'


# with open(JSON_all) as f :
#     data_all = json.load(f) 

# data_train = {}
# data_train["version"]=data_all["version"]
# data_train["date"]=data_all["date"]
# data_train["labels"]=data_all["labels"]
# data_train["videos"] = []

# data_val = {}
# data_val["version"]=data_all["version"]
# data_val["date"]=data_all["date"]
# data_val["labels"]=data_all["labels"]
# data_val["videos"] = []

# data_test = {}
# data_test["version"]=data_all["version"]
# data_test["date"]=data_all["date"]
# data_test["labels"]=data_all["labels"]
# data_test["videos"] = []

# for i, vid in enumerate(data_all["videos"]):
#     if i<38:
#         data_train["videos"].append(vid)
#     elif i<38+13:
#         data_val["videos"].append(vid)
#     else:
#         data_test["videos"].append(vid)

    
# with open(JSON_train, 'w') as f:
#     json.dump(data_train, f, indent=4)
# with open(JSON_test, 'w') as f:
#     json.dump(data_test, f, indent=4)
# with open(JSON_val, 'w') as f:
#     json.dump(data_val, f, indent=4)
    
