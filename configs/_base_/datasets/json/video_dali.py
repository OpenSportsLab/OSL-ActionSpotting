classes = 'soccernetv2/2_fps/dali/class.txt'
            
dataset = dict(
    batch_size = 8,
    epoch_num_frames  = 500000,
    mixup = True,
    modality = 'rgb',
    crop_dim = -1,
    dilate_len = 0,               # Dilate ground truth labels
    clip_len = 100,
    train=dict(
        type="VideoGameWithDali",
        classes=classes,
        output_map = ["data", "label"],
        label_file = "datasets/soccernetv2/2_fps/dali/train.json",               # path to label json
        data_root = "/home/ybenzakour/datasets/SoccerNet/",
        stride = 12,
        dataloader=dict(
            batch_size=8,
            shuffle=True,
        ),
    ),
    val=dict(
        type="VideoGameWithDali",
        classes=classes,
        output_map = ["data", "label"],
        stride = 12,
        label_file = "datasets/soccernetv2/2_fps/dali/val.json",                 # path to label json
        data_root = "/home/ybenzakour/datasets/SoccerNet/",
        dataloader=dict(
            batch_size=8,
            shuffle=True,
        ),
    ),
    val_data_frames=dict(
        type="VideoGameWithDaliVideo",
        classes=classes,
        stride = 12,
        output_map = ["data", "label"],
        label_file = "datasets/soccernetv2/2_fps/dali/val.json",                 # path to label json
        data_root = "/home/ybenzakour/datasets/SoccerNet/",
        overlap_len = 0,
        dataloader=dict(
            batch_size=4,
            shuffle=False,
        ),
    ),
    test = dict(
        type="VideoGameWithDaliVideo",
        stride = 12,
        classes=classes,
        output_map = ["data", "label"],
        label_file = "datasets/soccernetv2/2_fps/dali/test.json",  # path to label json
        data_root = "/home/ybenzakour/datasets/SoccerNet/",
        split=["test"],
        # results = "results_spotting_test",
        results = "pred-test.141.recall.json.gz",
        metric = "loose",
        overlap_len = 50,
        dataloader=dict(
            batch_size=4,
            shuffle=False,
        ),
    ),
    challenge = dict(
        type="VideoGameWithDaliVideo",
        overlap_len = 50,
        stride = 12,
        output_map = ["data", "label"],
        label_file = "datasets/soccernetv2/2_fps/dali/challenge.json",  # path to label json
        data_root = "/home/ybenzakour/datasets/SoccerNet/",
        dataloader=dict(
            batch_size=4,
            shuffle=False,
        ),
    )
)


