#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : 59-Lmq
# @Time     : 2023/3/26 10:49
# @File     : loss_dice.py
# @Project  : ToothSegmentation

import torch


def dice_acc(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)

    return loss


def dice_multi(score, target):
    num_class = score.shape[1]

    total_acc = 0
    total_acc_with_0 = 0
    for c in range(num_class):
        dice_acc_ = dice_acc(score[:, c, :, :], target[:, c, :, :])
        total_acc += dice_acc_
        if c != 0:
            total_acc_with_0 += dice_acc_

    total_acc /= num_class
    total_acc_with_0 /= num_class

    return total_acc, total_acc_with_0
