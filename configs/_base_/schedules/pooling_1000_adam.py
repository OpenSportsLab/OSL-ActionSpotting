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
