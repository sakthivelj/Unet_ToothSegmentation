#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : 59-Lmq
# @Time     : 2023/3/23 16:10
# @File     : loss.py
# @Project  : ToothSegmentation

import torch
import torch.nn as nn
import numpy as np
import random


def label_to_onehot(label, num_classes):
    """
    将多维标签转换为OneHot编码。

    Args:
        label: 多维标签张量，形状为 [batch_size, *spatial_dims]，数据类型为 torch.LongTensor。
        num_classes: 类别数。

    Returns:
        OneHot编码张量，形状为 [batch_size, num_classes, *spatial_dims]，数据类型为 torch.FloatTensor。
    """
    assert len(label.shape) >= 2, "标签张量必须是二维或以上的"

    batch_size = label.shape[0]
    spatial_dims = label.shape[1:]
    onehot = torch.zeros(batch_size, num_classes, *spatial_dims).to(label.device)
    label = label.unsqueeze(1)  # 将标签张量扩展一维，变成 [batch_size, 1, *spatial_dims]

    # 使用scatter_()函数将标签张量分散到对应位置
    onehot.scatter_(1, label, 1)

    return onehot


def weight_to_onehot(weight, label_shape):
    """
    将一维权重张量转换为和多维标签OneHot编码同纬度的张量。

    Args:
        weight: 一维权重张量，形状为 [batch_size * num_classes]，数据类型为 torch.FloatTensor。
        label_shape: 多维标签张量的形状，形状为 [batch_size, *spatial_dims]。

    Returns:
        和多维标签OneHot编码同纬度的张量，形状为 [batch_size, num_classes, *spatial_dims]，数据类型为 torch.FloatTensor。
    """
    assert len(label_shape) >= 2, "标签张量必须是二维或以上的"

    batch_size = label_shape[0]
    spatial_dims = label_shape[1:]
    num_classes = weight.shape[0] // batch_size

    # 将一维权重张量重塑为 [batch_size, num_classes, 1, 1, ...]
    weight = weight.reshape(batch_size, num_classes, *((1,) * len(spatial_dims)))

    # 创建全零张量作为输出
    onehot = torch.zeros(batch_size, num_classes, *spatial_dims).to(weight.device)

    # 使用scatter_()函数将权重张量分散到对应位置
    onehot.scatter_(1, torch.zeros_like(onehot).scatter_(1, torch.arange(num_classes).view(1, num_classes, *(
                (1,) * len(spatial_dims))).to(weight.device), 1), weight)

    return onehot

def to_one_hot(num_classes, label):
    """
    标签转独热编码
    :param num_classes: 类别数
    :param label: shape(batch_size, x, y, z) or (batch_size, h, w)
    :return: one_hot_label, shape(batch_size, num_classes x, y, z)
    """
    shape_ = label.shape  # [batch_size, x, y, z]
    if len(shape_) == 4:
        template_size = (shape_[0], shape_[1], shape_[2], shape_[3])  # [batch_size, x, y, z]
        template_size_view = (shape_[0], 1, shape_[1], shape_[2], shape_[3])  # [batch_size, 1, x, y, z]
    elif len(shape_) == 3:
        template_size = (shape_[0], shape_[1], shape_[2])  # [batch_size, h, w]
        template_size_view = (shape_[0], 1, shape_[1], shape_[2])  # [batch_size, 1, h, w]

    one_hots = []  # 记录one_hot标签

    for i in range(num_classes):
        template = torch.ones(template_size)
        template[label != i] = 0  # 在 label != 当前标签值的地方赋值为0
        template = template.view(template_size_view)

        one_hots.append(template)  # 存储当前标签
    one_hot_label = torch.cat(one_hots, dim=1)  # 所有标签的矩阵拼接起来

    return one_hot_label


def to_one_hot_alpha(num_classes, label, alpha):
    """
    标签转独热编码
    :param num_classes: 类别数
    :param label: shape(batch_size, x, y, z) or (batch_size, h, w)
    :param alpha: shape(num_classes, ), 权重值
    :return: one_hot_label, shape(batch_size, num_classes x, y, z)
    """
    shape_ = label.shape  # [batch_size, x, y, z]
    # print(f'label:\n{label}')

    if len(shape_) == 4:
        template_size = (shape_[0], shape_[1], shape_[2], shape_[3])  # [batch_size, x, y, z]
        template_size_view = (shape_[0], 1, shape_[1], shape_[2], shape_[3])  # [batch_size, 1, x, y, z]
    elif len(shape_) == 3:
        template_size = (shape_[0], shape_[1], shape_[2])  # [batch_size, h, w]
        template_size_view = (shape_[0], 1, shape_[1], shape_[2])  # [batch_size, 1, h, w]

    one_hots = []  # 记录one_hot标签
    alpha_one_hots = []  # 记录alpha的one hot标签

    for i in range(num_classes):
        template = torch.ones(template_size)
        template_a = torch.ones(template_size) * alpha[i]

        template[label != i] = 0  # 在 label != 当前标签值的地方赋值为0

        template = template.view(template_size_view)
        template_a = template_a.view(template_size_view)

        one_hots.append(template)  # 存储当前标签
        alpha_one_hots.append(template_a)

    one_hot_label = torch.cat(one_hots, dim=1)  # 所有标签的矩阵拼接起来
    one_hot_alpha = torch.cat(alpha_one_hots, dim=1)

    return one_hot_label, one_hot_alpha


class FocalLoss(nn.Module):
    def __init__(self,
                 alpha: list = [0.2, 0.3, 0.5],
                 gamma: float = 2.0,
                 num_class: int = 3,
                 reduction: str = 'mean',
                 device=None,
                 if_fl: bool = True):
        """
        注意，本 Focal Loss 输入的是已经softmax的outputs
        :param alpha: 权重系数列表，如三分类中第0类权重0.2，第1类权重0.3，第2类权重0.5
        :param gamma: 困难样本挖掘的gamma
        :param num_class: 用于计算的类别
        :param reduction:选择是计算均值还是和，'mean' or 'sum'
        :param device: 计算过程中的设备，输入时记得填写。
        :param if_fl:是否计算gamma部分，默认计算，即True
        """
        super(FocalLoss, self).__init__()

        assert len(alpha) == num_class
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.reduction = reduction
        self.device = device
        self.if_fl = if_fl

    def forward(self, prob, label):
        """
        迭代过程……
        :param prob: 输入的模型预测概率图，如
            output = net(x);
            prob = F.softmax(dim=1)，经过softmax之后的概率
            常见的三维shape是，(batch_size, num_class, x, y, z)或者二维shape:(batch_size, num_class, h, w)
        :param label: 输入的标签，常见三维shape：(batch_size, x, y, z)或则二维shape:(batch_size, h, w)

        output: [batch_size, num_class, 220, 220, 220]
        label: [batch_size, 220, 220, 220]
        one_hot_label: [batch_size, num_class, 220, 220, 220]

        alpha_t: shape:(num_class,),  Like :[0.2, 0.3, 0.5]
        probability: F.softmax(output), shape: [batch_size, 33, 220, 220, 220]

        alpha = [1, 220, 220, 220]
        """

        # 标签和权重都进行独热编码
        one_hot_label, alpha = to_one_hot_alpha(prob.shape[1], label, self.alpha)
        one_hot_label, alpha = one_hot_label.to(self.device), alpha.to(self.device)
        # one hot label shape:(batch_size, num_class, x, y, z); alpha's shape is the same

        # print(f'label:\n{label}')
        # print(f'label.shape:{label.shape}')
        #
        # print(f'one_hot_label:\n{one_hot_label}')
        # print(f'one_hot_label.shape:{one_hot_label.shape}')
        #
        # print(f'alpha:\n{alpha}')
        # print(f'alpha.shape:{alpha.shape}')

        # cross entropy，即softmax 交叉熵
        ce_loss = torch.mul(-torch.log(prob), one_hot_label)  # ce loss shape:(batch_size, num_class, x, y, z);
        # print(f'ce_loss:\n{ce_loss}')
        # print(f'ce_loss.shape:{ce_loss.shape}')

        """
        交叉熵公式， Loss = - label · log(softmax(outputs))
        α-balanced 交叉熵， Loss = - alpha · label · log(softmax(outputs))
        Focal Loss: Loss = - (1 - p)^γ · alpha · label · log(softmax(outputs))
        """
        loss = alpha * ce_loss  # 交叉熵 ✖ α，即α-balanced 交叉熵

        # multiply (1 - pt) ^ gamma，可以将focal loss的公式理解为：FL_Loss = (1 - p)^γ * CE_Loss
        if self.if_fl:
            loss = (torch.pow((1 - prob), self.gamma)) * loss

        # print(f'loss:\n{loss}')
        # print(f'loss.shape:{loss.shape}')

        loss = loss.sum(dim=1)
        # print(f'loss:\n{loss}')
        # print(f'loss.shape:{loss.shape}')
        if self.reduction == "mean":
            return torch.mean(loss)
        if self.reduction == "sum":
            return torch.sum(loss)
        return loss


class ClassBalancedLoss(nn.Module):
    def __init__(self,
                 gamma: float = 2,
                 beta: float = 0.999,
                 sample_per_class: list = [20, 75, 5],
                 num_class: int = 3,
                 reduction: str = 'mean',
                 device=None,
                 loss_type: str = 'focal'):
        """
        注意，本 ClassBalancedLoss 输入的是已经softmax的outputs
        Class Balanced Loss 在论文中的公式为： Loss = (1-β) / (1-β^n) * Focal Loss, where Focal Loss can be replaced by any Loss.
        其中 分母的 n 即是每个类别的有效样本数量，指未经过 augmentation 的数据总量。

        :param gamma: 困难样本挖掘的gamma，用于Focal Loss
        :param beta:
            Class Balanced Loss中原始参数，常用为0.9或0.999
            其在论文中的数学公式为：β = (N-1) / N, where N 是样本空间总体积，即样本总数量
        :param sample_per_class:
            意指用于Class Balanced Loss中的类别有效样本数量，
            其内列表的第 i 个元素代指第 i 个类别在原始数据集中的所有样本数量。
        :param num_class: 用于计算的类别
        :param reduction:选择是计算均值还是和，'mean' or 'sum'
        :param device: 计算过程中的设备，输入时记得填写。
        :param loss_type: 选择使用的损失函数，'focal'
        """
        super(ClassBalancedLoss, self).__init__()
        self.device = device
        self.gamma = gamma
        self.beta = beta
        self.num_class = num_class
        self.reduction = reduction
        self.sample_per_class = sample_per_class
        self.loss_type = loss_type

        self.__get_weights()  # 计算CB的权重系数

    def __get_weights(self):
        # 计算权重 (1-beta) / (1 - beta^n)
        effective_number = 1 - np.power(self.beta, self.sample_per_class)  # 分母,shape:(num_class,)
        # print(f' effective_number:{effective_number}')
        weights = (1.0 - self.beta) / np.array(effective_number)  # 整个权重 shape:(num_class,)
        # print(f'weights:{weights}')

        # 归一化后，再乘以类别总数，weight_i = (weight_i / sum(weights)) * num_class
        weights = weights / np.sum(weights) * self.num_class  # shape: (num_class,)
        # print(f' weights:{weights}')
        self.weights = weights

    def forward(self, prob, label):
        """
        迭代过程……
        :param prob: 输入的模型预测概率图，如
            output = net(x);
            prob = F.softmax(dim=1)，经过softmax之后的概率
            常见的三维shape是，(batch_size, num_class, x, y, z)或者二维shape:(batch_size, num_class, h, w)
        :param label: 输入的标签，常见三维shape：(batch_size, x, y, z)或则二维shape:(batch_size, h, w)

        output: [batch_size, 33, 220, 220, 220]
        label: [batch_size, 220, 220, 220]
        one_hot_label: [batch_size, 33, 220, 220, 220]

        alpha_t: [num_class], [0.2, 0.3, 0.5]
        probability: F.softmax(output), shape: [batch_size, 33, 220, 220, 220]

        alpha = [1, 220, 220, 220]

        """

        if self.loss_type == 'focal':
            FL_Criterion = FocalLoss(alpha=self.weights, gamma=self.gamma,
                                     num_class=self.num_class,
                                     device=self.device, if_fl=True)
            loss = FL_Criterion(prob, label)
        elif self.loss_type == 'ce_loss':
            self.weights = torch.tensor(self.weights).to(self.device)
            CE_Criterion = torch.nn.CrossEntropyLoss(weight=self.weights)
            loss = CE_Criterion(prob, label)

        return loss


class DiceLoss(nn.Module):
    def __init__(self,
                 smooth: float = 1e-5,
                 num_class: int = 33,
                 device=None):
        """
        多分类语义分割的DiceLoss
        :param smooth: Dice Loss的平滑系数
        :param num_class: 需要计算的类别数量
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.num_class = num_class
        self.device = device

    def forward(self, outputs, label):
        """
        :param outputs: 输入的网络预测outputs.shape:[batch_size, num_class, x, y, z]
        :param label: label.shape:[batch_size, x, y, z]
        """
        total_loss = 0
        C = self.num_class  # num_class 类别数
        label_one_hot = to_one_hot(C, label)  # 变成one_hot编码
        label_one_hot = label_one_hot.to(self.device)
        # print(f'label_one_hot: \n{label_one_hot}')

        del label

        if len(label_one_hot.shape) == 5:  # 输入是[batch_size, num_class, x, y, z]
            for c in range(C):
                intersect = torch.sum(outputs[:, c, :, :, :] * label_one_hot[:, c, :, :, :])
                outputs_sum = torch.sum(outputs[:, c, :, :, :] * outputs[:, c, :, :, :])
                labels_sum = torch.sum(label_one_hot[:, c, :, :, :] * label_one_hot[:, c, :, :, :])
                dice_loss = (2 * intersect + self.smooth) / (outputs_sum + labels_sum + self.smooth)
                dice_loss = 1 - dice_loss
                total_loss += dice_loss
        elif len(label_one_hot.shape) == 4:  # 输入是[batch_size, num_class, h, w]
            for c in range(C):
                intersect = torch.sum(outputs[:, c, :, :] * label_one_hot[:, c, :, :])
                outputs_sum = torch.sum(outputs[:, c, :, :] * outputs[:, c, :, :])
                labels_sum = torch.sum(label_one_hot[:, c, :, :] * label_one_hot[:, c, :, :])
                dice_loss = (2 * intersect + self.smooth) / (outputs_sum + labels_sum + self.smooth)
                # print(f'inter:{intersect}, y:{outputs_sum}, z:{labels_sum}, d_loss:{dice_loss}')
                dice_loss = 1 - dice_loss
                total_loss += dice_loss

        loss = total_loss / C

        # print(f'loss:{loss}\t')
        return loss


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def __test_dice_loss():
    set_random_seed(300)
    batch_size = 1
    x, y, z = (5, 5, 5)
    num_c = 3
    dl = DiceLoss(num_class=num_c)
    out = torch.rand(size=(batch_size, num_c, x, y, z))
    lab = torch.randint(0, num_c, size=(batch_size, x, y, z))
    print(f'out:\n{out}')
    print(f'lab:\n{lab}')
    loss = dl(out, lab)
    print(loss)


def __test_dice_loss_val():
    """
    input:
     i_0 = 0.4874+0.7601+0.4848+0.4432+0.7159
     i_1 = 0.2039+0.1400+0.3527+0.6558
     i = i_0+i_1
     i
    output: 4.2438

    input:
     y_sum_0 = 0.4874**2+0.4340**2+0.5817**2+0.7601**2+0.4848**2+0.4432**2+0.4386**2+0.1852**2+0.7159**2
     y_sum_1 = 0.2959**2+0.2039**2+0.1400**2+0.6402**2+0.3111**2+0.6042**2+0.3527**2+0.6558**2+0.5073**2
     y_sum = y_sum_0 + y_sum_1
     y_sum
    output: 4.34493388

    input:
     l_sum_0 = 1+1+1+1+1
     l_sum_1 = 1+1+1+1
     l_sum = l_sum_0+l_sum_1
     l_sum
    output: 9

    input:
     dice = (2*i * 1e-5) / (y_sum + l_sum+1e-5)
     dice
    output:  6.360161628495362e-06

    input:
     dice = (2*i + 1e-5) / (y_sum + l_sum+1e-5)
     dice
    output:  0.6360169121970111

    input:
     dice_0 = (2 * i_0 + 1e-5) / (y_sum_0 + l_sum_0 + 1e-5)
     dice_0
    output:  0.7697388582113539

    input:
     dice_1 = (2 * i_1 + 1e-5) / (y_sum_1 + l_sum_1 + 1e-5)
     dice_1
    output: 0.46376679853263075

    input:
     dice_average = (dice_0 + dice_1) / 2
     dice_average
    output:  0.6167528283719923
    """
    set_random_seed(300)
    batch_size = 1
    h, w = (3, 3)
    num_c = 2
    dl = DiceLoss(num_class=num_c)
    out = torch.rand(size=(batch_size, num_c, h, w))
    lab = torch.randint(0, num_c, size=(batch_size, h, w))
    print(f'out:\n{out}')
    print(f'lab:\n{lab}')
    loss = dl(out, lab)
    print(loss)  # output:1-0.6168 = 0.3832


def __test_focal_loss():
    set_random_seed(3407)
    num_c = 3
    alpha = [0.1 * i for i in range(num_c)]
    out = torch.rand(size=(1, num_c, 4, 4, 4))
    lab = torch.randint(0, num_c, size=(1, 4, 4, 4))
    fl = FocalLoss(alpha=alpha, device='cpu')
    loss = fl(out, lab)
    print(f'loss: {loss}')


def __test_focal_loss_2():
    set_random_seed(3407)
    num_c = 3
    alpha = [0.1 * i for i in range(num_c)]
    out = torch.rand(size=(1, num_c, 4, 4))
    lab = torch.randint(0, num_c, size=(1, 4, 4))
    fl = FocalLoss(alpha=alpha, device='cpu')
    loss = fl(out, lab)
    print(f'loss: {loss}')


def __test_class_balanced_focal_loss():
    def __get_sample_per_class_in_segmentation(labels):
        """
        获取语义分割中，每个类的有效样本数量，
        实际上，在做特征工程的时候，
        或者数据预处理的时候，就应该将训练集中的每个类的有效样本数量计算出来。

        """
        class_number = torch.unique(labels, return_counts=True)[1]
        return (class_number / torch.sum(class_number)) * 100

    set_random_seed(3407)
    batch_size = 10
    num_c = 3
    out = torch.rand(size=(batch_size, num_c, 22, 22, 22))
    lab = torch.randint(0, num_c, size=(batch_size, 22, 22, 22))
    # class_all_unique = torch.unique(lab, return_counts=True)  # tuple: (tensor(each label), tensor(each class number))
    # class_all = class_all_unique[1]  # obtain tensor(each class number)
    #
    # class_all = class_all / torch.sum(class_all)
    # class_all = 100 * class_all
    # sample_cls = (class_all_unique[1] / torch.sum(class_all_unique[1])) * 100

    sample_cls = __get_sample_per_class_in_segmentation(lab)
    # print(f'out:\n{out}')
    # print(f'lab:\n{lab}')

    fl = ClassBalancedLoss(sample_per_class=sample_cls, device='cpu')
    loss = fl(out, lab)
    print(f'loss: {loss}')


def __test_other_one_hot():
    set_random_seed(3407)
    batch_size = 1
    num_c = 2
    lab = torch.randint(0, num_c, size=(batch_size, 2, 2))
    ont_hot = torch.zeros(batch_size, num_c, 2, 2).scatter_(1, lab.view(batch_size, 1, 2, 2), 1)
    print(f'lab:\n{lab}')
    print(f'ont_hot:\n{ont_hot}')

    batch_size = 1
    num_c = 2
    lab = torch.randint(0, num_c, size=(batch_size, 2, 2, 2))
    alp = torch.ones_like(lab)
    print(f'alp:\n{alp}')
    alp = torch.tensor([0.2, 0.8]) * alp
    ont_hot = torch.zeros(batch_size, num_c, 2, 2, 2).scatter_(1, lab.view(batch_size, 1, 2, 2, 2), 1)
    print(f'alp:\n{alp}')
    # alp = alp[ont_hot.type(torch.long)]
    print(f'lab:\n{lab}')
    print(f'ont_hot:\n{ont_hot}')
    print(f'alp:\n{alp}')
    # print(f'alp:\n{alp * ont_hot}')


if __name__ == '__main__':
    # __test_dice_loss()
    # __test_dice_loss_val()
    # __test_focal_loss()
    # __test_class_balanced_focal_loss()
    __test_other_one_hot()
