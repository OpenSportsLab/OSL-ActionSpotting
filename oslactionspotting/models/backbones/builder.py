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
import torch
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import timm

from oslactionspotting.models.utils.shift import make_temporal_shift


def build_backbone(cfg, default_args=None):
    """Build a backbone from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        default_args (dict | None, optional): Default initialization arguments.
            Default: None.

    Returns:
        backbone: The constructed backbone.
    """
    if cfg.type == "PreExtactedFeatures":
        backbone = PreExtactedFeatures(
            feature_dim=cfg.feature_dim, output_dim=cfg.output_dim
        )
    elif cfg.type in ["rn18", "rn18_tsm", "rn18_gsm", "rn50", "rn50_tsm", "rn50_gsm"]:
        backbone = ResnetExtractFeatures(
            cfg.type, cfg.clip_len, cfg.is_rgb, cfg.in_channels
        )
    elif cfg.type in [
        "rny002",
        "rny002_tsm",
        "rny002_gsm",
        "rny008",
        "rny008_tsm",
        "rny008_gsm",
    ]:
        backbone = RegnetyExtractFeatures(
            cfg.type, cfg.clip_len, cfg.is_rgb, cfg.in_channels
        )
    elif cfg.type in ["convnextt", "convnextt_tsm", "convnextt_gsm"]:
        backbone = ConvNextTinyExtractFeatures(
            cfg.type, cfg.clip_len, cfg.is_rgb, cfg.in_channels
        )
    else:
        backbone = None

    return backbone


def Add_Temporal_Shift_Modules(
    feature_arch,
    features,
    clip_len,
):
    """Add temporal shift modules to a model.

    Args:
        feature_arch (string): name of the feature.
        features: The model.
        clip_len (int): The length of the clip.

    Returns:
        require_clip_len (int): The required length of clip.
    """
    require_clip_len = -1
    if feature_arch.endswith("_tsm"):
        make_temporal_shift(features, clip_len, is_gsm=False)
        require_clip_len = clip_len
    elif feature_arch.endswith("_gsm"):
        make_temporal_shift(features, clip_len, is_gsm=True)
        require_clip_len = clip_len
    return require_clip_len


class BaseExtractFeatures(nn.Module):
    """Base parent class for feature extractor model used by the E2E method.
    They all share the same forward method.
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        batch_size, true_clip_len, channels, height, width = inputs.shape

        clip_len = true_clip_len
        if self._require_clip_len > 0:
            assert (
                true_clip_len <= self._require_clip_len
            ), "Expected {}, got {}".format(self._require_clip_len, true_clip_len)
            if true_clip_len < self._require_clip_len:
                inputs = F.pad(
                    inputs, (0,) * 7 + (self._require_clip_len - true_clip_len,)
                )
                clip_len = self._require_clip_len

        im_feat = self._features(inputs.view(-1, channels, height, width)).reshape(
            batch_size, clip_len, self._feat_dim
        )

        if true_clip_len != clip_len:
            im_feat = im_feat[:, :true_clip_len, :]

        return im_feat


class ConvNextTinyExtractFeatures(BaseExtractFeatures):
    """Feature extractor which is based on the "convnext_tiny" of the timm models.
    The model is adapted for this task by adding temporal shift modules.

    Args:
        feature_arch (string): Feature extractor architecture.
        clip_len (int): Length of the clips.
        is_rgb (bool): Whether images are rgb or not.
        in_channels (int): Number of channels of images.
    """

    def __init__(self, feature_arch, clip_len, is_rgb, in_channels):
        super().__init__()
        features = timm.create_model("convnext_tiny", pretrained=is_rgb)
        feat_dim = features.head.fc.in_features
        features.head.fc = nn.Identity()

        if not is_rgb:
            features.stem[0] = nn.Conv2d(in_channels, 96, kernel_size=4, stride=4)

        # Add Temporal Shift Modules
        self._require_clip_len = Add_Temporal_Shift_Modules(
            feature_arch, features, clip_len
        )

        self._features = features
        self._feat_dim = feat_dim


class RegnetyExtractFeatures(BaseExtractFeatures):
    """Feature extractor which is based on the "regnet" models of the timm models.
    The model is adapted for this task by adding temporal shift modules.

    Args:
        feature_arch (string): Feature extractor architecture.
        clip_len (int): Length of the clips.
        is_rgb (bool): Whether images are rgb or not.
        in_channels (int): Number of channels of images.
    """

    def __init__(self, feature_arch, clip_len, is_rgb, in_channels):
        super().__init__()
        features = timm.create_model(
            {
                "rny002": "regnety_002",
                "rny008": "regnety_008",
            }[feature_arch.rsplit("_", 1)[0]],
            pretrained=is_rgb,
        )
        feat_dim = features.head.fc.in_features
        features.head.fc = nn.Identity()
        if not is_rgb:
            features.stem.conv = nn.Conv2d(
                in_channels,
                32,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                bias=False,
            )

        # Add Temporal Shift Modules
        self._require_clip_len = Add_Temporal_Shift_Modules(
            feature_arch, features, clip_len
        )

        self._features = features
        self._feat_dim = feat_dim


class ResnetExtractFeatures(nn.Module):
    """Feature extractor which is based on the "resnet" models of the torchvision models.
    The model is adapted for this task by adding temporal shift modules.

    Args:
        feature_arch (string): Feature extractor architecture.
        clip_len (int): Length of the clips.
        is_rgb (bool): Whether images are rgb or not.
        in_channels (int): Number of channels of images.
    """

    def __init__(self, feature_arch, clip_len, is_rgb, in_channels):
        super().__init__()

        resnet_name = feature_arch.split("_")[0].replace("rn", "resnet")
        features = getattr(torchvision.models, resnet_name)(pretrained=is_rgb)
        feat_dim = features.fc.in_features
        features.fc = nn.Identity()
        # import torchsummary
        # print(torchsummary.summary(features.to('cuda'), (3, 224, 224)))

        # Flow has only two input channels
        if not is_rgb:
            # FIXME: args maybe wrong for larger resnet
            features.conv1 = nn.Conv2d(
                in_channels,
                64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )
        # Add Temporal Shift Modules
        self._require_clip_len = Add_Temporal_Shift_Modules(
            feature_arch, features, clip_len
        )

        self._features = features
        self._feat_dim = feat_dim


class PreExtactedFeatures(torch.nn.Module):
    """Class used when processing features directly. This class is used to adapt dimensions of the inputs if they do not correspond between layers."""

    def __init__(self, feature_dim, output_dim):
        super(PreExtactedFeatures, self).__init__()
        self.feature_dim = feature_dim
        self.output_dim = output_dim

        self.reduceDim = not self.feature_dim == self.output_dim
        if self.reduceDim:
            self.feature_extractor = torch.nn.Linear(self.feature_dim, self.output_dim)

    def forward(self, inputs):
        if len(inputs.shape) == 4:
            BS, D2, FR, IC = inputs.shape
            if self.reduceDim:
                inputs = inputs.reshape(BS * FR * D2, IC)
                inputs = self.feature_extractor(inputs)
                inputs = inputs.reshape(BS, D2, FR, -1)
        else:
            BS, FR, IC = inputs.shape
            if self.reduceDim:
                inputs = inputs.reshape(BS * FR, IC)
                inputs = self.feature_extractor(inputs)
                inputs = inputs.reshape(BS, FR, -1)
        return inputs
