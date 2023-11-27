model = dict(
    type='LearnablePooling',
    load_weights=None,
    backbone=dict(
        type='PreExtactedFeatures',
        encoder='ResNET_TF2_PCA512',
        feature_dim=512,
        output_dim=512,
        framerate=2,
        window_size=20),
    neck=dict(
        type='NetVLAD++',
        input_dim=512,
        output_dim=512*64,
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
    type="trainer_pooling",
    max_epochs=1000,
    evaluation_frequency=1000,
    framerate=2,
    batch_size=256,
    GPU=0,
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
runner = dict(
    type="runner_pooling"
)


