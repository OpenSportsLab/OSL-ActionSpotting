model = dict(
    type='ContextAware',
    load_weights=None,
    input_size=512, 
    num_classes=17, 
    chunk_size=120, 
    dim_capsule=16,
    receptive_field=40, 
    num_detections=15, 
    framerate=2,
    # backbone=dict(
    #     type='PreExtactedFeatures',
    #     encoder='ResNET_TF2_PCA512',
    #     feature_dim=512,
    #     output_dim=512,
    #     framerate=2,
    #     window_size=20),
    # neck=dict(
    #     type='NetVLAD++',
    #     input_dim=512,
    #     output_dim=512*64,
    #     vocab_size=64),
    # head=dict(
    #     type='LinearLayer',
    #     input_dim=64*512,
    #     num_classes=17),
    # post_proc=dict(
    #     type="NMS",
    #     NMS_window=30,
    #     NMS_threshold=0.0),
)

training = dict(
    type="trainer_CALF",
    max_epochs=1000,
    evaluation_frequency=1000,
    framerate=2,
    batch_size=32,
    GPU=0,
    criterion = dict(
        type="Combined2x",
        w_1 = 0.000367,
        loss_1 = dict(
            type="ContextAwareLoss",
            K=[[-100, -98, -20, -40, -96, -5, -8, -93, -99, -31, -75, -10, -97, -75, -20, -84, -18], 
            [-50, -49, -10, -20, -48, -3, -4, -46, -50, -15, -37, -5, -49, -38, -10, -42, -9], 
            [50, 49, 60, 10, 48, 3, 4, 46, 50, 15, 37, 5, 49, 38, 10, 42, 9], 
            [100, 98, 90, 20, 96, 5, 8, 93, 99, 31, 75, 10, 97, 75, 20, 84, 18]],
            framerate=2,
            hit_radius = 0.1,
            miss_radius = 0.9
        ),
        w_2 = 1.0,
        loss_2 = dict(
            type="SpottingLoss",
            lambda_coord=5.0,
            lambda_noobj=0.5
        ),
    ),
    optimizer = dict(
        type="Adam",
        lr=1e-3, 
        betas=(0.9, 0.999), 
        eps=1e-07, 
        weight_decay=0, 
        amsgrad=False
    ),
    scheduler=dict(
        type="ReduceLROnPlateau",
        mode="min",
        LR=1e-03, 
        LRe=1e-06, 
        patience=25,
        verbose=True,
    ),
)
runner = dict(
    type="runner_CALF"
)
