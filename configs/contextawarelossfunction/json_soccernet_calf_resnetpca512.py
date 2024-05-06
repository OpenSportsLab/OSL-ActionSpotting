_base_ = [
    "../_base_/datasets/json/features_clips_CALF.py",  # dataset config
    "../_base_/models/contextawarelossfunction.py",  # model config
    "../_base_/schedules/calf_1000_adam.py", # trainer config
]

work_dir = "outputs/contextawarelossfunction/json_soccernet_calf_resnetpca512"

dataset = dict(
    train=dict(path='datasets_jsons/soccernetv2/features/Train.json'),
    val=dict(path="datasets_jsons/soccernetv2/features/Valid.json"),
    test=dict(path="/scratch/users/ybenzakour/zip/features/Test.json")
)
runner = dict(
    type="runner_JSON"
)