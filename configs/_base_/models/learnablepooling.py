model = dict(
    type='LearnablePooling',
    load_weights=None,
    backbone=dict(
        type='PreExtactedFeatures',
        feature_dim=512,
        output_dim=512,
        framerate=2,
        window_size=20),
    neck=dict(
        type='NetVLAD++',
        input_dim=512,
        vocab_size=64),
    head=dict(
        type='LinearLayer',
        input_dim=64*512,
        num_classes=17),
    post_proc=dict(
        type="NMS",
        NMS_window=30,
        NMS_threshold=0.0),
)

training = dict(
    max_epochs=1000, #1000,
    evaluation_frequency=10, #10,
    framerate=2,
    batch_size=256,
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
