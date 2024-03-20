classes = '/soccernetv2/2_fps/opencv/class.txt'
labels_dir = '/soccernetv2/2_fps/opencv/'
            
dataset = dict(
    batch_size = 8,
    epoch_num_frames  = 500000,
    mixup = True,
    modality = 'rgb',
    crop_dim = -1,
    dilate_len = 0,               # Dilate ground truth labels
    clip_len = 100,
    train=dict(
        type="VideoGameWithOpencv",
        classes=classes,
        label_file = "/soccernetv2/2_fps/opencv/train.json",                 # path to label json
        data_root = "/home/ybenzakour/datasets/SoccerNet/",
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
        label_file = "/soccernetv2/2_fps/opencv/val.json",                 # path to label json
        data_root = "/home/ybenzakour/datasets/SoccerNet/",
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
        label_file = "/soccernetv2/2_fps/opencv/val.json",                 # path to label json
        data_root = "/home/ybenzakour/datasets/SoccerNet/",
        dataloader=dict(
            num_workers=8,
            batch_size=4,
            shuffle=False,
            pin_memory=True,
            prefetch_factor = 1,
        ),
    ),
)


