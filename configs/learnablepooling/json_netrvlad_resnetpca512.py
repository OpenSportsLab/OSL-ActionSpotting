_base_ = [
    "../_base_/datasets/json/features_clips.py",  # dataset config
    "../_base_/models/learnablepooling.py",  # model config
    "../_base_/schedules/pooling_1000_adam.py" # trainer config
]

work_dir = "outputs/learnablepooling/json_netrvlad_resnetpca512"

log_level = 'INFO'  # The level of logging
model = dict(
    neck=dict(
        type='NetRVLAD',
        vocab_size=64),
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