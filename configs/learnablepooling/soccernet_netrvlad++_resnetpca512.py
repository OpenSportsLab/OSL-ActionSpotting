_base_ = [
    "../_base_/datasets/soccernet/features_clips.py",  # dataset config
    "../_base_/models/learnablepooling.py",  # model config
    "../_base_/schedules/pooling_1000_adam.py" # trainer config
]

work_dir = "outputs/learnablepooling/soccernet_netrvlad++_resnetpca512"

dataset = dict(
    train=dict(features="ResNET_TF2_PCA512.npy"),
    val=dict(features="ResNET_TF2_PCA512.npy"),
    test=dict(features="ResNET_TF2_PCA512.npy")
)

model = dict(
    neck=dict(
        type='NetRVLAD++',
        vocab_size=64),
)