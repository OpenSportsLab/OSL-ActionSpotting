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


runner = dict(
    type="runner_CALF"
)
