#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : 59-Lmq
# @Time     : 2023/3/29 16:18
# @File     : unet_senet.py
# @Project  : ToothSegmentation
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        y = self.avg_pool(x).view(batch_size, channels)
        y = self.fc(y).view(batch_size, channels, 1, 1)
        return x * y.expand_as(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.se1 = SEBlock(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se2 = SEBlock(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.se1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.se2(x)
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


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.se1 = SEBlock(128)
        self.down2 = Down(128, 256)
        self.se2 = SEBlock(256)
        self.down3 = Down(256, 512)
        self.se3 = SEBlock(512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.se4 = SEBlock(256)
        self.up2 = Up(512, 128)
        self.se5 = SEBlock(128)
        self.up3 = Up(256, 64)
        self.se6 = SEBlock(64)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.se1(x2)
        x3 = self.down2(x2)
        x3 = self.se2(x3)
        x4 = self.down3(x3)
        x4 = self.se3(x4)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.se4(x)
        x = self.up2(x, x3)
        x = self.se5(x)
        x = self.up3(x, x2)
        x = self.se6(x)
        x = self.outc(x)
        return x
