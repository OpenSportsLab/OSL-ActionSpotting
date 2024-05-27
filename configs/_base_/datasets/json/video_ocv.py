classes = 'datasets_jsons/soccernetv2/2_fps/dali/class.txt'
            
dataset = dict(
    epoch_num_frames  = 500000,
    mixup = True,
    modality = 'rgb',
    crop_dim = -1,
    dilate_len = 0,               # Dilate ground truth labels
    clip_len = 100,
    input_fps = 25,
    extract_fps = 2,
    train=dict(
        type="VideoGameWithOpencv",
        classes=classes,
        path = "/home/ybenzakour/224p/train/annotations.json",                 # path to label json
        data_root = "/home/ybenzakour/datasets/SoccerNet/",
        dataloader=dict(
            num_workers=8,
            batch_size=8,
            shuffle=False,
            pin_memory=True,
            prefetch_factor = 1,
        ),
    ),
    valid=dict(
        type="VideoGameWithOpencv",
        classes=classes,
        path = "/home/ybenzakour/224p/valid/annotations.json",                 # path to label json
        data_root = "/home/ybenzakour/datasets/SoccerNet/",
        dataloader=dict(
            num_workers=8,
            batch_size=8,
            shuffle=False,
            pin_memory=True,
            prefetch_factor = 1,
        ),
    ),
    valid_data_frames=dict(
        type="VideoGameWithOpencvVideo",
        classes=classes,
        path = "/home/ybenzakour/224p/valid/annotations.json",                 # path to label json
        data_root = "/home/ybenzakour/datasets/SoccerNet/",
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
        path = "/home/ybenzakour/224p/test/annotations.json",                 # path to label json
        data_root = "/home/ybenzakour/datasets/SoccerNet/",
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
        path = "/home/ybenzakour/224p/challenge/annotations.json",                 # path to label json
        data_root = "/home/ybenzakour/datasets/SoccerNet/",
        overlap_len = 50,
        dataloader=dict(
            num_workers=8,
            batch_size=4,
            shuffle=False,
            pin_memory=True,
        ),
    ),
)


