training = dict(
    type="trainer_e2e",
    num_epochs = 100,
    acc_grad_iter = 1,
    warm_up_epochs = 3,
    base_num_val_epochs = 20,
    start_val_epoch = None,
    criterion = 'map',
    learning_rate = 0.001,
    inference_batch_size = 4,
    GPU = 1
    )