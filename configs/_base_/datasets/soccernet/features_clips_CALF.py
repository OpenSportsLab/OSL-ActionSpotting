
classes = ["Penalty", "Kick-off", "Goal", "Substitution", "Offside", 
    "Shots on target", "Shots off target", "Clearance", "Ball out of play", 
    "Throw-in", "Foul", "Indirect free-kick", "Direct free-kick", "Corner", 
    "Yellow card","Red card", "Yellow->red card"]
data_root = '/home/ybenzakour/datasets/SoccerNet/'
dataset = dict(
    train=dict(
        type="SoccerNetClipsCALF",
        features="ResNET_TF2_PCA512.npy",
        version=2,
        framerate=2,
        chunk_size=120,
        receptive_field=40,
        chunks_per_epoch=6000,
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
    valid=dict(
        type="SoccerNetClipsCALF",
        features="ResNET_TF2_PCA512.npy",
        version=2,
        framerate=2,
        chunk_size=120,
        receptive_field=40,
        chunks_per_epoch=6000,
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
            shuffle=False,
            pin_memory=True,
        )),
    test=dict(
        type="SoccerNetClipsTestingCALF",
        features="ResNET_TF2_PCA512.npy",
        version=2,
        framerate=2,
        chunk_size=120,
        receptive_field=40,
        chunks_per_epoch=6000,
        split=["test"],
        data_root=data_root,
        pipeline=[
            dict(type='LoadFeatureFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
        ],
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
