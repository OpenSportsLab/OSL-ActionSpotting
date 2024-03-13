_base_ = [
    "../_base_/datasets/json/video_dali.py",  # dataset config
    "../_base_/models/e2espot.py",  # model config
]

work_dir = "outputs/e2e/rny002_gsm"

dali = True
num_epochs = 50
clip_len = 100
modality = 'rgb'
acc_grad_iter = 1
warm_up_epochs = 3
learning_rate = 0.001
start_val_epoch = None
criterion = 'map'
inference_batch_size = 4
training = dict(
    type="trainer_e2e")