model = dict(
    type='E2E',
    feature_arch = 'rny002_gsm',
    temporal_arch = 'gru',
    multi_gpu = False,
    load_weights = None,
)

runner = dict(
    type="runner_e2e"
)


