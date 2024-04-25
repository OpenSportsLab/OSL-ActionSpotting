import os
import re
import json
import pickle
import gzip

import torch

def load_json(fpath):
    with open(fpath) as fp:
        return json.load(fp)


def load_gz_json(fpath):
    with gzip.open(fpath, 'rt', encoding='ascii') as fp:
        return json.load(fp)


def store_json(fpath, obj, pretty=False):
    kwargs = {}
    if pretty:
        kwargs['indent'] = 2
        kwargs['sort_keys'] = True
    with open(fpath, 'w') as fp:
        json.dump(obj, fp, **kwargs)


def store_gz_json(fpath, obj):
    with gzip.open(fpath, 'wt', encoding='ascii') as fp:
        json.dump(obj, fp)


def load_pickle(fpath):
    with open(fpath, 'rb') as fp:
        return pickle.load(fp)


def store_pickle(fpath, obj):
    with open(fpath, 'wb') as fp:
        pickle.dump(obj, fp)


def load_text(fpath):
    lines = []
    with open(fpath, 'r') as fp:
        for l in fp:
            l = l.strip()
            if l:
                lines.append(l)
    return lines


def store_text(fpath, s):
    with open(fpath, 'w') as fp:
        fp.write(s)


def clear_files(dir_name, re_str, exclude=[]):
    for file_name in os.listdir(dir_name):
        if re.match(re_str, file_name):
            if file_name not in exclude:
                file_path = os.path.join(dir_name, file_name)
                os.remove(file_path)

def check_config(cfg):
    from oslspotting.core.utils.dataset import load_classes
    # check if cuda available
    has_gpu=torch.cuda.is_available()
    if 'GPU' in cfg.training.keys():
        if cfg.training.GPU >= 0:
            if not has_gpu:
                cfg.training.GPU = -1
    else :
        cfg.training.GPU = 1
    if cfg.runner.type == "runner_e2e":
        assert cfg.dataset.modality in ['rgb']
        assert cfg.model.backbone.type in [
                # From torchvision
                'rn18',
                'rn18_tsm',
                'rn18_gsm',
                'rn50',
                'rn50_tsm',
                'rn50_gsm',

                # From timm (following its naming conventions)
                'rny002',
                'rny002_tsm',
                'rny002_gsm',
                'rny008',
                'rny008_tsm',
                'rny008_gsm',

                # From timm
                'convnextt',
                'convnextt_tsm',
                'convnextt_gsm'
            ]
        assert cfg.model.head.type in ['', 'gru', 'deeper_gru', 'mstcn', 'asformer']
        assert cfg.dataset.batch_size % cfg.training.acc_grad_iter == 0
        assert cfg.training.criterion in ['map', 'loss']
        if cfg.training.start_val_epoch is None:
            cfg.training.start_val_epoch = cfg.training.num_epochs - cfg.training.base_num_val_epochs
        if cfg.dataset.crop_dim <= 0:
            cfg.dataset.crop_dim = None
        assert os.path.isfile(cfg.classes) and os.path.exists(cfg.classes)
        cfg.classes = load_classes(cfg.classes)
        for key,value in cfg.dataset.items():
            if key in ['train','val','val_data_frames','test','challenge']:
                pass
            else:
                cfg.dataset['train'][key] = value
                cfg.dataset['val'][key] = value
                cfg.dataset['val_data_frames'][key] = value
                cfg.dataset['test'][key] = value
                cfg.dataset['challenge'][key] = value
