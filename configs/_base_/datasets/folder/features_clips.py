# data_root = 'path/to/SoccerNet/'
classes = ("Medical",) 

dataset = dict(
    max_num_worker=4,
    train=dict(
        type="FolderClips",
        path="train.json",
        # features="ResNET_TF2_PCA512.npy",
        # version=2,
        framerate=2,
        window_size=20,
        # split=["train"],
        # data_root=data_root,
        # pipeline=[
        #     dict(type='LoadFeatureFromFile'),
        #     dict(type='LoadAnnotations', with_bbox=True),
        # ],
        classes=classes,
        dataloader=dict(
            num_workers=4,
            batch_size=256,
            shuffle=True,
            pin_memory=True,
        )),
    val=dict(
        type="FolderClips",
        path="val.json",
        # features="ResNET_TF2_PCA512.npy",
        # version=2,
        framerate=2,
        window_size=20,
        # split=["valid"],
        # data_root=data_root,
        # pipeline=[
        #     dict(type='LoadFeatureFromFile'),
        #     dict(type='LoadAnnotations', with_bbox=True),
        # ],
        classes=classes,
        dataloader=dict(
            num_workers=4,
            batch_size=256,
            shuffle=True,
            pin_memory=True,
        )),
    test=dict(
        type="FolderGames",
        path="test.json",
        # features="ResNET_TF2_PCA512.npy",
        # version=2,
        framerate=2,
        window_size=20,
        split=["test"],
        # data_root=data_root,
        # pipeline=[
        #     dict(type='LoadFeatureFromFile'),
        #     dict(type='LoadAnnotations', with_bbox=True),
        # ],
        classes=classes,
        dataloader=dict(
            num_workers=1,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
        )),
)
