classes = 'class.txt'

            
dataset = dict(
    epoch_num_frames  = 500000,
    batch_size = 8,
    mixup = True,
    dilate_len = 0,               # Dilate ground truth labels
    crop_dim = 224,
    train=dict(
        type="VideoGameWithDali",
        classes=classes,
        output_map = ["data", "label"],
        label_file = "train.json",                 # path to label json
    ),
    val=dict(
        type="VideoGameWithDali",
        classes=classes,
        output_map = ["data", "label"],
        label_file = "val.json",                 # path to label json
    ),
    val_data_frames=dict(
        type="VideoGameWithDaliVideo",
        classes=classes,
        output_map = ["data", "label"],
        label_file = "val.json",                 # path to label json
        ),
)


