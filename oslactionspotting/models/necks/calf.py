import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CNN_temporally_aware(torch.nn.Module):
    def __init__(
        self,
        input_size=512,
        num_classes=3,
        chunk_size=240,
        dim_capsule=16,
        receptive_field=80,
        num_detections=5,
        framerate=2,
    ):
        super(CNN_temporally_aware, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.dim_capsule = dim_capsule
        self.receptive_field = receptive_field
        self.num_detections = num_detections
        self.chunk_size = chunk_size
        self.framerate = framerate

        self.pyramid_size_1 = int(np.ceil(receptive_field / 7))
        self.pyramid_size_2 = int(np.ceil(receptive_field / 3))
        self.pyramid_size_3 = int(np.ceil(receptive_field / 2))
        self.pyramid_size_4 = int(np.ceil(receptive_field))

        # Base Convolutional Layers
        self.conv_1 = nn.Conv2d(
            in_channels=1, out_channels=128, kernel_size=(1, input_size)
        )
        self.conv_2 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 1))

        # Temporal Pyramidal Module
        self.pad_p_1 = nn.ZeroPad2d(
            (
                0,
                0,
                (self.pyramid_size_1 - 1) // 2,
                self.pyramid_size_1 - 1 - (self.pyramid_size_1 - 1) // 2,
            )
        )
        self.pad_p_2 = nn.ZeroPad2d(
            (
                0,
                0,
                (self.pyramid_size_2 - 1) // 2,
                self.pyramid_size_2 - 1 - (self.pyramid_size_2 - 1) // 2,
            )
        )
        self.pad_p_3 = nn.ZeroPad2d(
            (
                0,
                0,
                (self.pyramid_size_3 - 1) // 2,
                self.pyramid_size_3 - 1 - (self.pyramid_size_3 - 1) // 2,
            )
        )
        self.pad_p_4 = nn.ZeroPad2d(
            (
                0,
                0,
                (self.pyramid_size_4 - 1) // 2,
                self.pyramid_size_4 - 1 - (self.pyramid_size_4 - 1) // 2,
            )
        )
        self.conv_p_1 = nn.Conv2d(
            in_channels=32, out_channels=8, kernel_size=(self.pyramid_size_1, 1)
        )
        self.conv_p_2 = nn.Conv2d(
            in_channels=32, out_channels=16, kernel_size=(self.pyramid_size_2, 1)
        )
        self.conv_p_3 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=(self.pyramid_size_3, 1)
        )
        self.conv_p_4 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(self.pyramid_size_4, 1)
        )

        # -------------------
        # Segmentation module
        # -------------------

        self.kernel_seg_size = 3
        self.pad_seg = nn.ZeroPad2d(
            (
                0,
                0,
                (self.kernel_seg_size - 1) // 2,
                self.kernel_seg_size - 1 - (self.kernel_seg_size - 1) // 2,
            )
        )
        self.conv_seg = nn.Conv2d(
            in_channels=152,
            out_channels=dim_capsule * num_classes,
            kernel_size=(self.kernel_seg_size, 1),
        )
        self.batch_seg = nn.BatchNorm2d(
            num_features=self.chunk_size, momentum=0.01, eps=0.001
        )

    def forward(self, inputs):
        # -------------------------------------
        # Temporal Convolutional neural network
        # -------------------------------------

        # Base Convolutional Layers
        conv_1 = F.relu(self.conv_1(inputs))
        # print("Conv_1 size: ", conv_1.size())

        conv_2 = F.relu(self.conv_2(conv_1))
        # print("Conv_2 size: ", conv_2.size())

        # Temporal Pyramidal Module
        conv_p_1 = F.relu(self.conv_p_1(self.pad_p_1(conv_2)))
        # print("Conv_p_1 size: ", conv_p_1.size())
        conv_p_2 = F.relu(self.conv_p_2(self.pad_p_2(conv_2)))
        # print("Conv_p_2 size: ", conv_p_2.size())
        conv_p_3 = F.relu(self.conv_p_3(self.pad_p_3(conv_2)))
        # print("Conv_p_3 size: ", conv_p_3.size())
        conv_p_4 = F.relu(self.conv_p_4(self.pad_p_4(conv_2)))
        # print("Conv_p_4 size: ", conv_p_4.size())

        concatenation = torch.cat((conv_2, conv_p_1, conv_p_2, conv_p_3, conv_p_4), 1)
        # print("Concatenation size: ", concatenation.size())

        # -------------------
        # Segmentation module
        # -------------------

        conv_seg = self.conv_seg(self.pad_seg(concatenation))
        # print("Conv_seg size: ", conv_seg.size())

        conv_seg_permuted = conv_seg.permute(0, 2, 3, 1)
        # print("Conv_seg_permuted size: ", conv_seg_permuted.size())

        conv_seg_reshaped = conv_seg_permuted.view(
            conv_seg_permuted.size()[0],
            conv_seg_permuted.size()[1],
            self.dim_capsule,
            self.num_classes,
        )
        # print("Conv_seg_reshaped size: ", conv_seg_reshaped.size())

        # conv_seg_reshaped_permuted = conv_seg_reshaped.permute(0,3,1,2)
        # print("Conv_seg_reshaped_permuted size: ", conv_seg_reshaped_permuted.size())

        conv_seg_norm = torch.sigmoid(self.batch_seg(conv_seg_reshaped))
        # print("Conv_seg_norm: ", conv_seg_norm.size())

        # conv_seg_norm_permuted = conv_seg_norm.permute(0,2,3,1)
        # print("Conv_seg_norm_permuted size: ", conv_seg_norm_permuted.size())

        output_segmentation = torch.sqrt(
            torch.sum(torch.square(conv_seg_norm - 0.5), dim=2) * 4 / self.dim_capsule
        )
        # print("Output_segmentation size: ", output_segmentation.size())

        return conv_seg, output_segmentation
