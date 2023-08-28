_base_ = [
    "../_base_/datasets/soccernet/features_clips.py",  # dataset config
    "../_base_/models/learnablepooling.py",  # model config
]

work_dir = "outputs/learnablepooling/soccernet_avgpool++_resnetpca512"

dataset = dict(
    train=dict(features="ResNET_TF2_PCA512.npy"),
    val=dict(features="ResNET_TF2_PCA512.npy"),
    test=dict(features="ResNET_TF2_PCA512.npy")
)

model = dict(
    neck=dict(type='AvgPool++', output_dim=512, nb_frames=20*2,),
    head=dict(input_dim=2*512)
)