# This file was modified from https://github.com/BobLiu20/YOLOv3_PyTorch
# It needed to be modified in order to accomodate for different strides in the
from __future__ import division
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F


class SACBlock(nn.Module):
    def __init__(self, inplanes, expand1x1_planes, bn_d=0.1):
        super(SACBlock, self).__init__()
        self.inplanes = inplanes
        self.bn_d = bn_d

        self.attention_x = nn.Sequential(
            nn.Conv2d(3, 9 * self.inplanes, kernel_size=7, padding=3),
            nn.BatchNorm2d(9 * self.inplanes, momentum=0.1),
        )

        self.position_mlp_2 = nn.Sequential(
            nn.Conv2d(9 * self.inplanes, self.inplanes, kernel_size=1),
            nn.BatchNorm2d(self.inplanes, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.inplanes, momentum=0.1),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        xyz = input[0]
        new_xyz = input[1]
        feature = input[2]
        N, C, H, W = feature.size()

        new_feature = F.unfold(feature, kernel_size=3, padding=1).view(N, -1, H, W)
        attention = F.sigmoid(self.attention_x(new_xyz))
        new_feature = new_feature * attention
        new_feature = self.position_mlp_2(new_feature)
        fuse_feature = new_feature + feature

        return xyz, new_xyz, fuse_feature


# ******************************************************************************

# number of layers per model
model_blocks = {
    21: [1, 1, 2, 2, 1],
    53: [1, 2, 8, 8, 4],
}


class Backbone_pt(nn.Module):
    """
       Class for DarknetSeg. Subclasses PyTorch's own "nn" module
    """

    def __init__(self):
        super(Backbone_pt, self).__init__()
        self.use_rgb = True
        self.use_depth = True
        self.use_xyz = True
        self.drop_prob = 0.01
        self.bn_d = 0.01
        self.OS = 16
        self.layers = 21
        print("Using squeezesegv3" + str(self.layers) + " Backbone")
        self.input_depth = 0
        self.input_idxs = []
        if self.use_rgb:
            self.input_depth += 3
            self.input_idxs.extend([0, 1, 2])
        if self.use_depth:
            self.input_depth += 1
            self.input_idxs.append(3)
        if self.use_xyz:
            self.input_depth += 3
            self.input_idxs.extend([4, 5, 6])
        print("Depth of backbone input = ", self.input_depth)

        self.strides = [2, 2, 2, 2, 1]

        current_os = 1
        for s in self.strides:
            current_os *= s
        print("Original OS: ", current_os)

        if self.OS > current_os:
            print("Can't do OS, ", self.OS,
                  " because it is bigger than original ", current_os)
        else:

            for i, stride in enumerate(reversed(self.strides), 0):
                if int(current_os) != self.OS:
                    if stride == 2:
                        current_os /= 2
                        self.strides[-1 - i] = 1
                    if int(current_os) == self.OS:
                        break
            print("New OS: ", int(current_os))
            print("Strides: ", self.strides)

        assert self.layers in model_blocks.keys()

        self.blocks = model_blocks[self.layers]

        self.conv1 = nn.Conv2d(self.input_depth, 32, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32, momentum=self.bn_d)
        self.relu1 = nn.LeakyReLU(0.1)

        self.enc1 = self._make_enc_layer(SACBlock, [32, 64], self.blocks[0],
                                         stride=self.strides[0], DS=True, bn_d=self.bn_d)
        self.enc2 = self._make_enc_layer(SACBlock, [64, 128], self.blocks[1],
                                         stride=self.strides[1], DS=True, bn_d=self.bn_d)
        self.enc3 = self._make_enc_layer(SACBlock, [128, 256], self.blocks[2],
                                         stride=self.strides[2], DS=True, bn_d=self.bn_d)
        self.enc4 = self._make_enc_layer(SACBlock, [256, 512], self.blocks[3],
                                         stride=self.strides[3], DS=True, bn_d=self.bn_d)
        self.enc5 = self._make_enc_layer(SACBlock, [512, 512], self.blocks[4],
                                         stride=self.strides[4], DS=False, bn_d=self.bn_d)

        self.dropout = nn.Dropout2d(self.drop_prob)

        self.last_channels = 512

        self.fc5 = nn.Linear(512 * 12 * 16, 512)

        self.bn5 = nn.BatchNorm1d(512)

    def _make_enc_layer(self, block, planes, blocks, stride, DS, bn_d=0.1):
        layers = []

        inplanes = planes[0]
        for i in range(0, blocks):
            layers.append(("residual_{}".format(i),
                           block(inplanes, planes, bn_d)))
        if DS == True:
            layers.append(("conv", nn.Conv2d(planes[0], planes[1],
                                             kernel_size=3,
                                             stride=2, dilation=1,
                                             padding=1, bias=False)))
            layers.append(("bn", nn.BatchNorm2d(planes[1], momentum=bn_d)))
            layers.append(("relu", nn.LeakyReLU(0.1)))

        return nn.Sequential(OrderedDict(layers))

    def run_layer(self, xyz, feature, layer, skips, os, flag=True):
        new_xyz = xyz
        if flag == True:
            xyz, new_xyz, y = layer[:-3]([xyz, new_xyz, feature])
            y = layer[-3:](y)
            xyz = F.upsample_bilinear(xyz, size=[xyz.size()[2] // 2, xyz.size()[3] // 2])
        else:
            xyz, new_xyz, y = layer([xyz, new_xyz, feature])
        if y.shape[2] < feature.shape[2] or y.shape[3] < feature.shape[3]:
            skips[os] = feature.detach()
            os *= 2
        feature = self.dropout(y)
        return xyz, feature, skips, os

    def forward(self, feature):
        skips = {}
        os = 1
        xyz = feature[:, 4:7, :, :]
        feature = self.relu1(self.bn1(self.conv1(feature)))

        xyz, feature, skips, os = self.run_layer(xyz, feature, self.enc1, skips, os)
        xyz, feature, skips, os = self.run_layer(xyz, feature, self.enc2, skips, os)
        xyz, feature, skips, os = self.run_layer(xyz, feature, self.enc3, skips, os)
        xyz, feature, skips, os = self.run_layer(xyz, feature, self.enc4, skips, os)
        xyz, feature, skips, os = self.run_layer(xyz, feature, self.enc5, skips, os, flag=False)

        feature = feature.view(feature.size(0), -1)

        feature = self.fc5(feature)
        feature = self.bn5(feature)

        # return feature, skips

        return feature

    def get_last_depth(self):
        return self.last_channels

    def get_input_depth(self):
        return self.input_depth
