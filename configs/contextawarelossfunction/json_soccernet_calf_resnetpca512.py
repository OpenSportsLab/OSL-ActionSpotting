_base_ = [
    "../_base_/datasets/json/features_clips_CALF.py",  # dataset config
    "../_base_/models/contextawarelossfunction.py",  # model config
    "../_base_/schedules/calf_1000_adam.py", # trainer config
]

work_dir = "outputs/contextawarelossfunction/json_soccernet_calf_resnetpca512"

dataset = dict(
    train=dict(path="path/to/SoccerNet/train.json"),
    val=dict(path="path/to/SoccerNet/val.json"),
    test=dict(path="path/to/SoccerNet/test.json")
)

runner = dict(
    type="runner_JSON"
)