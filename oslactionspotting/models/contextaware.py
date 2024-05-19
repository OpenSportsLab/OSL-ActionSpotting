import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from oslactionspotting.models.litebase import LiteBaseModel

import os

from oslactionspotting.datasets.utils import timestamps2long, batch2long

from oslactionspotting.models.utils import (
    NMS,
    check_if_should_predict,
    get_json_data,
    predictions2json,
    predictions2json_runnerjson,
    zipResults,
)

from .heads import build_head
from .backbones import build_backbone
from .necks import build_neck


class ContextAwareModel(nn.Module):
    """
    CALF model composed of a backbone, neck and head.
    Args:
        weights (string): Path of the weights file.
        backbone (string): Name of the backbone type.
        neck (string): Name of the neck type.
        head (string): Name of the head type.
    The model takes as input a Tensor of the form (batch_size,1,chunk_size,input_size)
    and returns :
        1. The segmentation of the form (batch_size,chunk_size,num_classes).
        2. The action spotting of the form (batch_size,num_detections,2+num_classes).
    """

    def __init__(
        self,
        weights=None,
        backbone="PreExtracted",
        neck="CNN++",
        head="SpottingCALF",
        post_proc="NMS",
    ):

        super(ContextAwareModel, self).__init__()

        # Build Backbone
        self.backbone = build_backbone(backbone)

        # Build Neck
        self.neck = build_neck(neck)

        # Build Head
        self.head = build_head(head)

        # load weight if needed
        self.load_weights(weights=weights)

    def load_weights(self, weights=None):
        if weights is not None:
            print("=> loading checkpoint '{}'".format(weights))
            checkpoint = torch.load(weights)
            self.load_state_dict(checkpoint["state_dict"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    weights, checkpoint["epoch"]
                )
            )

    def forward(self, inputs):
        """
        INPUT: a Tensor of the form (batch_size,1,chunk_size,input_size)
        OUTPUTS:    1. The segmentation of the form (batch_size,chunk_size,num_classes)
                    2. The action spotting of the form (batch_size,num_detections,2+num_classes)
        """
        features = self.backbone(inputs)
        conv_seg, output_segmentation = self.neck(features)
        output_spotting = self.head(conv_seg, output_segmentation)
        return output_segmentation, output_spotting


class LiteContextAwareModel(LiteBaseModel):
    """
    Lightning module for the CALF model.
    Args:
        cfg (dict): DIct of config.
        weights (string): Path of the weights file.
        backbone (string): Name of the backbone type for the CALF model.
        neck (string): Name of the neck type for the CALF model.
        head (string): Name of the head type for the CALF model.
        runner (string): Name of the runner. "runner_CALF" if using SoccerNet dataset modules or "runner_JSON" if using the json format. This will the change the behaviour of processing the predictions while infering.
    """

    def __init__(
        self,
        cfg=None,
        weights=None,
        backbone="PreExtracted",
        neck="CNN++",
        head="SpottingCALF",
        post_proc="NMS",
        runner="runner_CALF",
    ):
        super().__init__(cfg.training)

        # check compatibility dims Backbone - Neck - Head
        assert backbone.output_dim == neck.input_size
        assert neck.num_classes == head.num_classes
        assert neck.dim_capsule == head.dim_capsule
        assert neck.num_detections == head.num_detections
        assert neck.chunk_size == head.chunk_size

        self.chunk_size = neck.chunk_size
        self.receptive_field = neck.receptive_field
        self.framerate = neck.framerate

        self.model = ContextAwareModel(weights, backbone, neck, head, post_proc)

        self.overwrite = True

        self.cfg = cfg

        self.runner = runner

        self.infer_split = getattr(cfg, "infer_split", True)

    def process(self, labels, targets, feats):
        labels = labels.float()
        targets = targets.float()
        feats = feats.unsqueeze(1)
        return labels, targets, feats

    def _common_step(self, batch, batch_idx):
        """Operations in common for training and validation steps.
        Process the features, labels and targets. The features are processed by the model to compute the outputs.
        These outputs are used to compute the loss.
        """
        feats, labels, targets = batch
        labels, targets, feats = self.process(labels, targets, feats)
        output_segmentation, output_spotting = self.forward(feats)
        return self.criterion(
            [labels, targets], [output_segmentation, output_spotting]
        ), feats.size(0)

    def training_step(self, batch, batch_idx):
        """Training step that defines the train loop."""
        loss, size = self._common_step(batch, batch_idx)
        self.log_dict({"loss": loss}, on_step=True, on_epoch=True, prog_bar=True)
        self.losses.update(loss.item(), size)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step that defines the val loop."""
        val_loss, size = self._common_step(batch, batch_idx)
        self.log_dict(
            {"val_loss": val_loss}, on_step=False, on_epoch=True, prog_bar=True
        )
        self.losses.update(val_loss.item(), size)
        return val_loss

    def on_predict_start(self):
        """Operations to make before starting to infer."""
        self.stop_predict = False

        if self.infer_split:
            self.output_folder, self.output_results, self.stop_predict = check_if_should_predict(
                self.cfg.dataset.test.results, self.cfg.work_dir, self.overwrite
            )
            if self.runner == "runner_JSON":
                self.target_dir = os.path.join(self.cfg.work_dir, self.output_folder)
            else:
                self.target_dir = self.output_results

        if not self.stop_predict:
            self.spotting_predictions = list()
            self.spotting_grountruth = list()
            self.spotting_grountruth_visibility = list()
            self.segmentation_predictions = list()

    def on_predict_end(self):
        """Operations to make after inference.
        The process is different whether the data come from json or from the SoccerNet dataset in the way we will store the jsons containing the predictions.
        """
        if not self.stop_predict:
            # Transformation to numpy for evaluation
            targets_numpy = list()
            closests_numpy = list()
            detections_numpy = list()
            for target, detection in zip(
                self.spotting_grountruth_visibility, self.spotting_predictions
            ):
                target_numpy = target.cpu().numpy()
                targets_numpy.append(target_numpy)
                detections_numpy.append(NMS(detection.numpy(), 20 * self.framerate))
                closest_numpy = np.zeros(target_numpy.shape) - 1
                # Get the closest action index
                for c in np.arange(target_numpy.shape[-1]):
                    indexes = np.where(target_numpy[:, c] != 0)[0].tolist()
                    if len(indexes) == 0:
                        continue
                    indexes.insert(0, -indexes[0])
                    indexes.append(2 * closest_numpy.shape[0])
                    for i in np.arange(len(indexes) - 2) + 1:
                        start = max(0, (indexes[i - 1] + indexes[i]) // 2)
                        stop = min(
                            closest_numpy.shape[0], (indexes[i] + indexes[i + 1]) // 2
                        )
                        closest_numpy[start:stop, c] = target_numpy[indexes[i], c]
                closests_numpy.append(closest_numpy)

            # Save the predictions to the json format
            # if save_predictions:
            if self.runner == "runner_CALF":
                list_game = self.trainer.predict_dataloaders.dataset.listGames
                for index in np.arange(len(list_game)):
                    json_data = get_json_data(list_game[index])
                    if self.infer_split:
                        os.makedirs(
                            os.path.join(
                                self.cfg.work_dir, self.output_folder, list_game[index]
                            ),
                            exist_ok=True,
                        )
                        output_file = os.path.join(
                            self.cfg.work_dir,
                            self.output_folder,
                            list_game[index],
                            "results_spotting.json",
                        )
                    else:
                        output_file = os.path.join(
                            self.cfg.work_dir, f"{self.cfg.dataset.test.results}.json"
                        )
                    json_data = predictions2json(
                        detections_numpy[index * 2],
                        detections_numpy[(index * 2) + 1],
                        json_data,
                        output_file,
                        self.framerate,
                    )
                    self.json_data = json_data
            elif self.runner == "runner_JSON":
                list_videos = self.trainer.predict_dataloaders.dataset.data_json[0][
                    "videos"
                ]
                for index in np.arange(len(list_videos)):
                    video = list_videos[index]["path_features"]

                    if self.infer_split:
                        video = os.path.splitext(video)[0]
                        os.makedirs(
                            os.path.join(self.cfg.work_dir, self.output_folder, video),
                            exist_ok=True,
                        )
                        output_file = os.path.join(
                            self.cfg.work_dir,
                            self.output_folder,
                            video,
                            "results_spotting.json",
                        )
                    else:
                        output_file = os.path.join(
                            self.cfg.work_dir, f"{self.cfg.dataset.test.results}.json"
                        )

                    json_data = get_json_data(video)
                    json_data = predictions2json_runnerjson(
                        detections_numpy[index],
                        json_data,
                        output_file,
                        self.framerate,
                        inverse_event_dictionary=self.trainer.predict_dataloaders.dataset.inverse_event_dictionary,
                    )
                    self.json_data = json_data
            if self.infer_split:
                zipResults(
                    zip_path=self.output_results,
                    target_dir=os.path.join(self.cfg.work_dir, self.output_folder),
                    filename="results_spotting.json",
                )

    def predict_step(self, batch):
        """Infer step.
        The process is different whether the data come from json or from the SoccerNet dataset.
        In particular, processing data from json means processing one video (features) while processing data from SOccerNet
        means processing two halfs of a game.
        """
        if not self.stop_predict:
            if self.runner == "runner_CALF":
                feat_half1, feat_half2, label_half1, label_half2 = batch

                label_half1 = label_half1.float().squeeze(0)
                label_half2 = label_half2.float().squeeze(0)

                feat_half1 = feat_half1.squeeze(0)
                feat_half2 = feat_half2.squeeze(0)

                feat_half1 = feat_half1.unsqueeze(1)
                feat_half2 = feat_half2.unsqueeze(1)

                # Compute the output
                output_segmentation_half_1, output_spotting_half_1 = self.forward(
                    feat_half1
                )
                output_segmentation_half_2, output_spotting_half_2 = self.forward(
                    feat_half2
                )

                timestamp_long_half_1 = timestamps2long(
                    output_spotting_half_1.cpu().detach(),
                    label_half1.size()[0],
                    self.chunk_size,
                    self.receptive_field,
                )
                timestamp_long_half_2 = timestamps2long(
                    output_spotting_half_2.cpu().detach(),
                    label_half2.size()[0],
                    self.chunk_size,
                    self.receptive_field,
                )
                segmentation_long_half_1 = batch2long(
                    output_segmentation_half_1.cpu().detach(),
                    label_half1.size()[0],
                    self.chunk_size,
                    self.receptive_field,
                )
                segmentation_long_half_2 = batch2long(
                    output_segmentation_half_2.cpu().detach(),
                    label_half2.size()[0],
                    self.chunk_size,
                    self.receptive_field,
                )

                self.spotting_grountruth.append(torch.abs(label_half1))
                self.spotting_grountruth.append(torch.abs(label_half2))
                self.spotting_grountruth_visibility.append(label_half1)
                self.spotting_grountruth_visibility.append(label_half2)
                self.spotting_predictions.append(timestamp_long_half_1)
                self.spotting_predictions.append(timestamp_long_half_2)
                self.segmentation_predictions.append(segmentation_long_half_1)
                self.segmentation_predictions.append(segmentation_long_half_2)
            elif self.runner == "runner_JSON":
                features, labels = batch

                labels = labels.float().squeeze(0)

                features = features.squeeze(0)

                features = features.unsqueeze(1)

                # Compute the output
                output_segmentation, output_spotting = self.forward(features)

                timestamp_long = timestamps2long(
                    output_spotting.cpu().detach(),
                    labels.size()[0],
                    self.chunk_size,
                    self.receptive_field,
                )
                segmentation_long = batch2long(
                    output_segmentation.cpu().detach(),
                    labels.size()[0],
                    self.chunk_size,
                    self.receptive_field,
                )

                self.spotting_grountruth.append(torch.abs(labels))
                self.spotting_grountruth_visibility.append(labels)
                self.spotting_predictions.append(timestamp_long)
                self.segmentation_predictions.append(segmentation_long)
