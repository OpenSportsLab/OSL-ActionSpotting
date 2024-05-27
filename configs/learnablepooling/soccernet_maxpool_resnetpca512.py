_base_ = [
    "../_base_/datasets/soccernet/features_clips.py",  # dataset config
    "../_base_/models/learnablepooling.py",  # model config
    "../_base_/schedules/pooling_1000_adam.py" # trainer config
]

work_dir = "outputs/learnablepooling/soccernet_maxpool_resnetpca512"

dataset = dict(
    train=dict(features="ResNET_TF2_PCA512.npy"),
    valid=dict(features="ResNET_TF2_PCA512.npy"),
    test=dict(features="ResNET_TF2_PCA512.npy")
)
log_level = 'INFO'  # The level of logging
model = dict(
    neck=dict(type='MaxPool', output_dim=512, nb_frames=20*2,),
    head=dict(input_dim=512)
)