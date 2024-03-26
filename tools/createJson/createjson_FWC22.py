import json
import os
import glob


data = {}
data["version"]=1
data["date"]="2023.08.31"
data["videos"] = []

DIR="/media/giancos/LaCie/FWC22/WC22/WC22/2022"
for fold in sorted(os.listdir(DIR)):
    # print(fold)
    label_file = os.path.join(DIR, fold,"Labels-medical.json")
    with open(label_file) as f :
        label = json.load(f) 
        # print(label)
    video = {}
    
    video_file = glob.glob(f'/media/giancos/LaCie/FWC22/*{fold}*.mp4')[0]
    video_file = os.path.basename(video_file)

    video["path_video"] = video_file
    video["path_features"] = fold+".npy"
    video["annotations"] = label["annotations"]

    data["videos"].append(video)

labels = [anno["label"] for v in data["videos"] for anno in v["annotations"] ]
data["labels"]=list(set(labels))
JSON_all = '/media/giancos/LaCie/FWC22/all.json'
with open(JSON_all, 'w') as f:
    json.dump(data, f, indent=4)



JSON_train = '/media/giancos/LaCie/FWC22/train.json'
JSON_val = '/media/giancos/LaCie/FWC22/val.json'
JSON_test = '/media/giancos/LaCie/FWC22/test.json'


with open(JSON_all) as f :
    data_all = json.load(f) 

data_train = {}
data_train["version"]=data_all["version"]
data_train["date"]=data_all["date"]
data_train["labels"]=data_all["labels"]
data_train["videos"] = []

data_val = {}
data_val["version"]=data_all["version"]
data_val["date"]=data_all["date"]
data_val["labels"]=data_all["labels"]
data_val["videos"] = []

data_test = {}
data_test["version"]=data_all["version"]
data_test["date"]=data_all["date"]
data_test["labels"]=data_all["labels"]
data_test["videos"] = []

for i, vid in enumerate(data_all["videos"]):
    if i<38:
        data_train["videos"].append(vid)
    elif i<38+13:
        data_val["videos"].append(vid)
    else:
        data_test["videos"].append(vid)

    
with open(JSON_train, 'w') as f:
    json.dump(data_train, f, indent=4)
with open(JSON_test, 'w') as f:
    json.dump(data_test, f, indent=4)
with open(JSON_val, 'w') as f:
    json.dump(data_val, f, indent=4)
    
