data_root = '/home/ybenzakour/datasets/SoccerNet/'
classes = ("Penalty", "Kick-off", "Goal", "Substitution", "Offside", 
    "Shots on target", "Shots off target", "Clearance", "Ball out of play", 
    "Throw-in", "Foul", "Indirect free-kick", "Direct free-kick", "Corner", 
    "Yellow card","Red card", "Yellow->red card",) 

dataset = dict(
    train=dict(
        type="SoccerNetClips",
        features="ResNET_TF2_PCA512.npy",
        version=2,
        framerate=2,
        window_size=20,
        split=["train"],
        data_root=data_root,
        pipeline=[
            dict(type='LoadFeatureFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
        ],
        classes=classes,
        dataloader=dict(
            num_workers=4,
            batch_size=256,
            shuffle=True,
            pin_memory=True,
        )),
    val=dict(
        type="SoccerNetClips",
        features="ResNET_TF2_PCA512.npy",
        version=2,
        framerate=2,
        window_size=20,
        split=["valid"],
        data_root=data_root,
        pipeline=[
            dict(type='LoadFeatureFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
        ],
        classes=classes,
        dataloader=dict(
            num_workers=4,
            batch_size=256,
            shuffle=True,
            pin_memory=True,
        )),
    test=dict(
        type="SoccerNetGames",
        features="ResNET_TF2_PCA512.npy",
        version=2,
        framerate=2,
        window_size=20,
        split=["test","challenge"],
        data_root=data_root,
        pipeline=[
            dict(type='LoadFeatureFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
        ],
        classes=classes,
        dataloader=dict(
            num_workers=1,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
        )),
)
