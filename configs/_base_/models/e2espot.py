model = dict(
    type='E2E',
    backbone = dict(
        type = 'rny008_gsm',
    ),
    head = dict(
        type = 'gru',
    ),
    # feature_arch = 'rny002_gsm',
    # temporal_arch = 'gru',
    multi_gpu = False,
    load_weights = None,
)

runner = dict(
    type="runner_e2e"
)


