data_root = 'path/to/SoccerNet/'
classes = ("Penalty", "Kick-off", "Goal", "Substitution", "Offside", 
    "Shots on target", "Shots off target", "Clearance", "Ball out of play", 
    "Throw-in", "Foul", "Indirect free-kick", "Direct free-kick", "Corner", 
    "Yellow card","Red card", "Yellow->red card",) 


model = dict(
    type='LearnablePooling',
    load_weights=None,
    backbone=dict(
        type='PreExtacted',
        feature_dim=512),
    neck='NetVLAD++',
    head='LinearLayer',
    classes=classes,
    framerate=2,
    window_size=20,
    vocab_size=64,
    NMS_window= 30,
    NMS_threshold= 0.0, 
)
dataset = dict(
    max_num_worker=4,
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
        type="SoccerNetClipsTesting",
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
training = dict(
    max_epochs=1000,
    evaluation_frequency=10,
    framerate=2,
    batch_size=256,
    LR=1e-03, 
    LRe=1e-06, 
    patience=10,
    GPU=-1,
    criterion = dict(
        type="NLLLoss",
    ),
    optimizer = dict(
        type="Adam",
        lr=1e-3, 
        betas=(0.9, 0.999), 
        eps=1e-08, 
        weight_decay=0, 
        amsgrad=False
    ),
    scheduler=dict(
        type="ReduceLROnPlateau",
        mode="min",
        LR=1e-03, 
        LRe=1e-06, 
        patience=10,
        verbose=True,
    ),
)
