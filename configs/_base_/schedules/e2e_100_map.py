training = dict(
    type="trainer_e2e",
    num_epochs = 100,
    acc_grad_iter = 1,
    base_num_val_epochs = 20,
    start_val_epoch = None,
    criterion_val = 'map',
    inference_batch_size = 4,
    GPU = 1,
    criterion = dict(
        type="CrossEntropyLoss",
    ),
    optimizer = dict(
        type="AdamWithScaler",
        learning_rate = 0.001, 
    ),
    scheduler=dict(
        type="ChainedSchedulerE2E",
        acc_grad_iter = 1,
        num_epochs = 100,
        warm_up_epochs = 3,
    ),
    )