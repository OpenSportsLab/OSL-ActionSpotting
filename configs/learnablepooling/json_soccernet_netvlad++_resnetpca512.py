_base_ = [
    "../_base_/datasets/json/features_clips.py",  # dataset config
    "../_base_/models/learnablepooling.py",  # model config
    "../_base_/schedules/pooling_1000_adam.py", # trainer config
]

work_dir = "outputs/learnablepooling/json_soccernet_netvlad++_resnetpca512"

dataset = dict(
    train=dict(path="/home/ybenzakour/datasets/SoccerNet/train.json"),
    val=dict(path="/home/ybenzakour/datasets/SoccerNet/val.json"),
    test=dict(path="/home/ybenzakour/datasets/SoccerNet/test.json")
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