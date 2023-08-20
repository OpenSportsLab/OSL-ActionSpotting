custom_imports = dict(
    imports=['backbones.tpcnn.py',
             'necks.segmentation.py',
             'heads.spotting.py',
             'losses.contextAwareLoss.py'],
    allow_failed_imports=False)


model = dict(
    backbone=dict(
        type='