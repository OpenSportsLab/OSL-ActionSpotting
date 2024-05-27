_base_ = [
    "../_base_/datasets/json/features_clips.py",  # dataset config
    "../_base_/models/learnablepooling.py",  # model config
    "../_base_/schedules/pooling_1000_adam.py", # trainer config
]

work_dir = "outputs/learnablepooling/json_soccernet_netvlad++_resnetpca512"

dataset = dict(
    train=dict(path='datasets_jsons/soccernetv2/features/Train.json'),
    valid=dict(path="datasets_jsons/soccernetv2/features/Valid.json"),
    test=dict(path="datasets_jsons/soccernetv2/features/Test.json")
)
log_level = 'INFO'  # The level of logging
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

visualizer = dict(
    threshold=0.0,
    annotation_range=5000,  # ms
    seconds_to_skip=30,
    scale=1.5,
)