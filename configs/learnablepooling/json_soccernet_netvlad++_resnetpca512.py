_base_ = [
    "../_base_/datasets/json/features_clips.py",  # dataset config
    "../_base_/models/learnablepooling.py",  # model config
    "../_base_/schedules/pooling_1000_adam.py", # trainer config
]

work_dir = "outputs/learnablepooling/json_soccernet_netvlad++_resnetpca512"

dataset = dict(
    train=dict(path='/scratch/users/ybenzakour/zip/features/Train.json'),
    val=dict(path="/scratch/users/ybenzakour/zip/features/Valid.json"),
    test=dict(path="/scratch/users/ybenzakour/zip/features/Test.json")
)

model = dict(
    neck=dict(
        type='NetVLAD++',
        vocab_size=64),
    head=dict(
        num_classes=17),
)

runner = dict(
    type="runner_JSON"
)