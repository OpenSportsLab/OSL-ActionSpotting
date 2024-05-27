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
        type="VideoGameWithDali",
        classes=classes,
        output_map = ["data", "label"],
        path = "/home/ybenzakour/224p/train/annotations.json",               # path to label json
        data_root = "/home/ybenzakour/datasets/SoccerNet/",
        dataloader=dict(
            batch_size=8,
            shuffle=True,
        ),
    ),
    valid=dict(
        type="VideoGameWithDali",
        classes=classes,
        output_map = ["data", "label"],
        path = "/home/ybenzakour/224p/valid/annotations.json",                 # path to label json
        data_root = "/home/ybenzakour/datasets/SoccerNet/",
        dataloader=dict(
            batch_size=8,
            shuffle=True,
        ),
    ),
    valid_data_frames=dict(
        type="VideoGameWithDaliVideo",
        classes=classes,
        output_map = ["data", "label"],
        path = "/home/ybenzakour/224p/valid/annotations.json",                 # path to label json
        data_root = "/home/ybenzakour/datasets/SoccerNet/",
        overlap_len = 0,
        dataloader=dict(
            batch_size=4,
            shuffle=False,
        ),
    ),
    test = dict(
        type="VideoGameWithDaliVideo",
        classes=classes,
        output_map = ["data", "label"],
        path = "/home/ybenzakour/224p/test/annotations.json",  # path to label json
        data_root = "/home/ybenzakour/datasets/SoccerNet/",
        split=["test"],
        results = "results_spotting_test",
        # results = "pred-test.141.recall.json.gz",
        nms_window = 2,
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
        output_map = ["data", "label"],
        path = "/home/ybenzakour/224p/challenge/annotations.json",  # path to label json
        data_root = "/home/ybenzakour/datasets/SoccerNet/",
        dataloader=dict(
            batch_size=4,
            shuffle=False,
        ),
    )
)


