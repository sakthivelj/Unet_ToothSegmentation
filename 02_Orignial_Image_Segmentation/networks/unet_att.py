#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : 59-Lmq
# @Time     : 2023/3/25 20:28
# @File     : unet_att.py
# @Project  : ToothSegmentation

import torch
import torch.nn as nn


class Unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super(Unet, self).__init__()

        # 下采样部分
        self.conv1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 上采样部分
        self.up1 = Up(512, 256)
        self.att1 = SelfAttention(256)
        self.up2 = Up(256, 128)
        self.att2 = SelfAttention(128)
        self.up3 = Up(128, 64)
        self.att3 = SelfAttention(64)
        self.up4 = Up(64, 64)
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # 下采样部分
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool1(x1))
        x3 = self.conv3(self.pool2(x2))
        x4 = self.conv4(self.pool3(x3))

        # 上采样部分
        x = self.up1(x4, x3)
        x = self.att1(x) * x
        x = self.up2(x, x2)
        x = self.att2(x) * x
        x = self.up3(x, x1)
        x = self.att3(x) * x
        x = self.up4(x)

        return self.out(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, height, width = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, height * width)
        energy = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(energy, dim=-1)
        proj_value = self.value
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, height, width)
        out = self.gamma * out + x

        return out


