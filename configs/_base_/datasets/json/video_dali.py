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
    extension = extension,
    train=dict(
        type="VideoGameWithDali",
        classes=classes,
        output_map = ["data", "label"],
        path = "datasets_jsons/soccernetv2/2_fps/dali/train.json",               # path to label json
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
        path = "datasets_jsons/soccernetv2/2_fps/dali/val.json",                 # path to label json
        data_root = "/home/ybenzakour/datasets/SoccerNet/",
        extension = extension,
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
        path = "datasets_jsons/soccernetv2/2_fps/dali/val.json",                 # path to label json
        data_root = "/home/ybenzakour/datasets/SoccerNet/",
        extension = extension,
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
        path = "datasets_jsons/soccernetv2/2_fps/dali/test.json",  # path to label json
        data_root = "/home/ybenzakour/datasets/SoccerNet/",
        extension = extension,
        split=["test"],
        results = "results_spotting_infer",
        # results = "pred-test.141.recall.json.gz",
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
        path = "datasets_jsons/soccernetv2/2_fps/dali/challenge.json",  # path to label json
        data_root = "/home/ybenzakour/datasets/SoccerNet/",
        extension = extension,
        dataloader=dict(
            batch_size=4,
            shuffle=False,
        ),
    )
)


