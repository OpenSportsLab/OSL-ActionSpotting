classes = ("Medical",) 

dataset = dict(
    train=dict(
        type="FeatureClipsfromJSON",
        path="train.json",
        framerate=2,
        window_size=20,
        classes=classes,
        dataloader=dict(
            num_workers=4,
            batch_size=256,
            shuffle=True,
            pin_memory=True,
        )),
    val=dict(
        type="FeatureClipsfromJSON",
        path="val.json",
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
