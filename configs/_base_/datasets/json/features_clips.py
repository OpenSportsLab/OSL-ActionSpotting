# classes = ("Medical",) 
classes = [
    "Penalty",
    "Kick-off",
    "Goal",
    "Substitution",
    "Offside",
    "Shots on target",
    "Shots off target",
    "Clearance",
    "Ball out of play",
    "Throw-in",
    "Foul",
    "Indirect free-kick",
    "Direct free-kick",
    "Corner",
    "Yellow card",
    "Red card",
    "Yellow->red card",
]
# classes = 'datasets_jsons/soccernetv2/features/class.txt'
# classes = '/scratch/users/ybenzakour/zip/features/class.txt'
# classes = '/home/ybenzakour/datasets/SoccerNet/class.txt'
dataset = dict(
    input_fps = 25,
    extract_fps = 2,
    train=dict(
        type="FeatureClipsfromJSON",
        path="train.json",
        data_root = "/home/ybenzakour/datasets/SoccerNet/",
        framerate=2,
        window_size=20,
        classes=classes,
        dataloader=dict(
            num_workers=4,
            batch_size=256,
            shuffle=True,
            pin_memory=True,
        )),
    valid=dict(
        type="FeatureClipsfromJSON",
        path="val.json",
        data_root = "/home/ybenzakour/datasets/SoccerNet/",
        framerate=2,
        window_size=20,
        classes=classes,
        dataloader=dict(
            num_workers=4,
            batch_size=256,
            shuffle=True,
            pin_memory=True,
        )),
    test=dict(
        type="FeatureVideosfromJSON",
        path="test.json",
        data_root = "/home/ybenzakour/datasets/SoccerNet/",
        framerate=2,
        window_size=20,
        split=["test"],
        classes=classes,
        metric = "loose",
        results = "results_spotting_test",
        dataloader=dict(
            num_workers=1,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
        )),
)
