_base_ = [
    "../_base_/datasets/json/video_dali.py",  # dataset config
    "../_base_/models/e2espot.py",  # model config,
    "../_base_/schedules/e2e_100_map.py", #trainer config
]

work_dir = "outputs/e2e/rny008_gsm_150"

dali = True