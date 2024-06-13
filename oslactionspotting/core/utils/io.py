"""
Copyright 2022 James Hong, Haotian Zhang, Matthew Fisher, Michael Gharbi,
Kayvon Fatahalian

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import os
import re
import json
import pickle
import gzip

import torch

from oslactionspotting.datasets.utils import get_repartition_gpu


def load_json(fpath):
    with open(fpath) as fp:
        return json.load(fp)


def load_gz_json(fpath):
    with gzip.open(fpath, "rt", encoding="ascii") as fp:
        return json.load(fp)


def store_json(fpath, obj, pretty=False):
    kwargs = {}
    if pretty:
        kwargs["indent"] = 4
        kwargs["sort_keys"] = False
    with open(fpath, "w") as fp:
        json.dump(obj, fp, **kwargs)


def store_gz_json(fpath, obj):
    with gzip.open(fpath, "wt", encoding="ascii") as fp:
        json.dump(obj, fp)


def load_text(fpath):
    """Load text from a given file.

    Args:
        fpath (string): The path of the file.

    Returns:
        lines (List): List in which element is a line of the file.

    """
    lines = []
    with open(fpath, "r") as fp:
        for l in fp:
            l = l.strip()
            if l:
                lines.append(l)
    return lines


def clear_files(dir_name, re_str, exclude=[]):
    for file_name in os.listdir(dir_name):
        if re.match(re_str, file_name):
            if file_name not in exclude:
                file_path = os.path.join(dir_name, file_name)
                os.remove(file_path)


def load_classes(input):
    """Load classes from either list or txt file.

    Args:
        input (string): Path of the file that contains one class per line or list of classes.

    Returns:
        Dictionnary with classes associated to indexes.
    """
    if isinstance(input, list):
        return {x: i + 1 for i, x in enumerate(sorted(input))}
    return {x: i + 1 for i, x in enumerate(load_text(input))}


def check_config(cfg):
    """Check for incoherences, missing elements in dict config.
    The checks are different regarding the methods.

    Args:
        cfg (dict): Config dictionnary.

    """
    # check if cuda available
    has_gpu = torch.cuda.is_available()
    if "GPU" in cfg.training.keys():
        if cfg.training.GPU >= 0:
            if not has_gpu:
                cfg.training.GPU = -1
    else:
        cfg.training.GPU = 1
    if cfg.runner.type == "runner_e2e":
        assert "dali" in cfg.keys()
        if cfg.dali == True:
            cfg.repartitions = get_repartition_gpu()
        assert cfg.dataset.modality in ["rgb"]
        assert cfg.model.backbone.type in [
            # From torchvision
            "rn18",
            "rn18_tsm",
            "rn18_gsm",
            "rn50",
            "rn50_tsm",
            "rn50_gsm",
            # From timm (following its naming conventions)
            "rny002",
            "rny002_tsm",
            "rny002_gsm",
            "rny008",
            "rny008_tsm",
            "rny008_gsm",
            # From timm
            "convnextt",
            "convnextt_tsm",
            "convnextt_gsm",
        ]
        assert cfg.model.head.type in ["", "gru", "deeper_gru", "mstcn", "asformer"]
        # assert cfg.dataset.batch_size % cfg.training.acc_grad_iter == 0
        assert cfg.dataset.train.dataloader.batch_size % cfg.training.acc_grad_iter == 0
        assert cfg.training.criterion_valid in ["map", "loss"]
        assert cfg.training.num_epochs == cfg.training.scheduler.num_epochs
        assert cfg.training.acc_grad_iter == cfg.training.scheduler.acc_grad_iter
        if cfg.training.start_valid_epoch is None:
            cfg.training.start_valid_epoch = (
                cfg.training.num_epochs - cfg.training.base_num_valid_epochs
            )
        if cfg.dataset.crop_dim <= 0:
            cfg.dataset.crop_dim = None

        if (
            cfg.dataset.test.path != None
            and os.path.isfile(cfg.dataset.test.path)
            and cfg.dataset.test.path.endswith(".json")
            and "labels" in load_json(cfg.dataset.test.path).keys()
        ):
            classes = load_json(cfg.dataset.test.path)["labels"]
        else:
            assert isinstance(cfg.classes, list) or os.path.isfile(cfg.classes)
            classes = cfg.classes
        cfg.classes = load_classes(classes)
        for key, value in cfg.dataset.items():
            if key in ["train", "valid", "valid_data_frames", "test", "challenge"]:
                pass
            else:
                cfg.dataset["train"][key] = value
                cfg.dataset["valid"][key] = value
                cfg.dataset["valid_data_frames"][key] = value
                cfg.dataset["test"][key] = value
                cfg.dataset["challenge"][key] = value
    else:
        for key, value in cfg.dataset.items():
            if key in ["train", "valid", "test"]:
                pass
            else:
                cfg.dataset["train"][key] = value
                cfg.dataset["valid"][key] = value
                cfg.dataset["test"][key] = value


def whether_infer_split(cfg):
    """Given a config dict, check whether we want to infer a split or a single element (can be a game, video or feature file)/

    Args:
        cfg (dict): Config dict.

    Returns:
        bool : True if we infer split, false otherwise. Raises an error if the input is not expected.
    """
    if cfg.type == "SoccerNetGames" or cfg.type == "SoccerNetClipsTestingCALF":
        if cfg.split == None:
            return False
        else:
            return True
    elif (
        cfg.type == "FeatureVideosfromJSON" or cfg.type == "FeatureVideosChunksfromJson"
    ):
        if cfg.path.endswith(".json"):
            return True
        else:
            return False
    elif cfg.type == "VideoGameWithOpencvVideo" or cfg.type == "VideoGameWithDaliVideo":
        if cfg.path.endswith(".json"):
            return True
        else:
            return False
    else:
        raise ValueError
