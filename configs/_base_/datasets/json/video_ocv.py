classes = 'datasets_jsons/soccernetv2/2_fps/dali/class.txt'
extension = '.mkv'
            
dataset = dict(
    batch_size = 8,
    epoch_num_frames  = 500000,
    mixup = True,
    modality = 'rgb',
    crop_dim = -1,
    dilate_len = 0,               # Dilate ground truth labels
    clip_len = 100,
    input_fps = 25,
    extract_fps = 2,
    extension = extension,
    train=dict(
        type="VideoGameWithOpencv",
        classes=classes,
        path = "datasets_jsons/soccernetv2/2_fps/opencv/Train.json",                 # path to label json
        data_root = "/home/ybenzakour/datasets/SoccerNet/",
        stride = 12,
        dataloader=dict(
            num_workers=8,
            batch_size=8,
            shuffle=False,
            pin_memory=True,
            prefetch_factor = 1,
        ),
    ),
    val=dict(
        type="VideoGameWithOpencv",
        classes=classes,
        path = "datasets_jsons/soccernetv2/2_fps/opencv/Valid.json",                 # path to label json
        data_root = "/home/ybenzakour/datasets/SoccerNet/",
        stride = 12,
        dataloader=dict(
            num_workers=8,
            batch_size=8,
            shuffle=False,
            pin_memory=True,
            prefetch_factor = 1,
        ),
    ),
    val_data_frames=dict(
        type="VideoGameWithOpencvVideo",
        classes=classes,
        path = "datasets_jsons/soccernetv2/2_fps/opencv/Valid.json",                 # path to label json
        data_root = "/home/ybenzakour/datasets/SoccerNet/",
        stride = 12,
        overlap_len = 0,
        dataloader=dict(
            num_workers=8,
            batch_size=4,
            shuffle=False,
            pin_memory=True,
            prefetch_factor = 1,
        ),
    ),
    test = dict(
        type="VideoGameWithOpencvVideo",
        classes=classes,
        path = "datasets_jsons/soccernetv2/2_fps/opencv/Test.json",                 # path to label json
        data_root = "/home/ybenzakour/datasets/SoccerNet/",
        stride = 12,
        split=["test"],
        results = "results_spotting_test_ocv",
        nms_window = 2,
        metric = "loose",
        overlap_len = 50,
        dataloader=dict(
            num_workers=8,
            batch_size=8,
            shuffle=False,
            pin_memory=True,
        ),
    ),
    challenge = dict(
        type="VideoGameWithOpencvVideo",
        classes=classes,
        path = "datasets_jsons/soccernetv2/2_fps/opencv/Challenge.json",                 # path to label json
        data_root = "/home/ybenzakour/datasets/SoccerNet/",
        stride = 12,
        overlap_len = 50,
        dataloader=dict(
            num_workers=8,
            batch_size=4,
            shuffle=False,
            pin_memory=True,
        ),
    ),
)


