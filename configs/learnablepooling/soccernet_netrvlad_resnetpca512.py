_base_ = [
    "../_base_/datasets/soccernet/features_clips.py",  # dataset config
    "../_base_/models/learnablepooling.py",  # model config
]

work_dir = "outputs/learnablepooling/soccernet_netrvlad_resnetpca512"

dataset = dict(
    train=dict(features="ResNET_TF2_PCA512.npy"),
    val=dict(features="ResNET_TF2_PCA512.npy"),
    test=dict(features="ResNET_TF2_PCA512.npy")
)

model = dict(
    neck='NetRVLAD',
)