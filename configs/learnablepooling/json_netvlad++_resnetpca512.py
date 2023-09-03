_base_ = [
    "../_base_/datasets/json/features_clips.py",  # dataset config
    "../_base_/models/learnablepooling.py",  # model config
]

work_dir = "outputs/learnablepooling/json_netvlad++_resnetpca512"

dataset = dict(
    train=dict(path="/media/giancos/LaCie/FWC22/train.json"),
    val=dict(path="/media/giancos/LaCie/FWC22/val.json"),
    test=dict(path="/media/giancos/LaCie/FWC22/test.json")
)

model = dict(
    neck=dict(
        type='NetVLAD++',
        vocab_size=64),
    head=dict(
        num_classes=1),
)

runner = dict(
    type="runner_JSON"
)