#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : 59-Lmq
# @Time     : 2023/3/23 16:50
# @File     : train.py
# @Project  : ToothSegmentation

# 基本库
import os
import sys
import time
import random
import shutil  # 复制代码
import argparse  # 初始设置
import logging  # 记录日志
from loguru import logger  # 记录日志

from tqdm import tqdm  # 控制循环鸽
import numpy as np

from tensorboardX import SummaryWriter  # 记录训练过程中的Loss和准确率

# torch 基础设置
import torch
import torch.optim as optim  # 优化器
from torchvision import transforms  # 数据预处理
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

# 导入网络
from networks.unet import Unet
import segmentation_models_pytorch as smp
from monai.networks.nets import AttentionUnet

# 导入损失函数
# from utils.loss import FocalLoss, DiceLoss, ClassBalancedLoss
from utils.loss_dice import dice_multi
from medpy import metric
from monai.losses import DiceLoss, FocalLoss, DiceCELoss, DiceFocalLoss

# 导入 data_loader
from dataloaders.toothLoader import *


def get_args():
    description = """
    测试smp的Unet, monai的Loss进行测试，目前是交叉熵损失函数
    使用Unet进行 原图 的33分类分割，loss 使用 CELoss 和 DiceLoss，batch_size统一使用1，最大迭代次数为500
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='..', help='Name of Experiment')
    parser.add_argument('--exp', type=str, default='exp_2023_03_31_09_multi_U-net_dl', help='实验名称')
    parser.add_argument('--description', type=str,
                        default=description,
                        help='实验描述')
    parser.add_argument('--if_original', type=bool, default=True, help='是否使用原图')
    parser.add_argument('--exp_which', type=str, default='DiceLoss',
                        help='FocalLoss, DiceLoss, DiceCELoss, DiceFocalLoss')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')

    parser.add_argument('--max_iterations', type=int, default=2000, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=3, help='batch_size per gpu')

    parser.add_argument('--use_state_model', type=bool, default=False, help='是否用之前的模型')
    parser.add_argument('--state_model_path', type=str, default='', help='之前保存模型的全路径')

    parser.add_argument('--num_class', type=int, default=33, help='class of you want to segment')

    parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--seed', type=int, default=3407, help='random seed')
    parser.add_argument('--num_workers', type=int, default=0, help='num-workers to use')
    args_ = parser.parse_args()

    return args_


def main():
    args = get_args()
    snapshot_path = "../experiments/" + args.exp + "/"

    log_path = os.path.join(snapshot_path, 'log/train_')
    logger.add(log_path + '{time}.log', rotation='00:00')

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    batch_size = args.batch_size * len(args.gpu.split(','))
    # print(batch_size)
    max_iterations = args.max_iterations
    base_lr = args.base_lr

    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    num_classes = args.num_class

    data_path = r"F:\pythonProject\Datasets\TeethData\2D\RT_17\HandedROI\data_list"
    img_path = r'F:\pythonProject\Datasets\TeethData\2D\RT_17\HandedROI\image'
    lab_path = r'F:\pythonProject\Datasets\TeethData\2D\RT_17\HandedROI\label'
    data_dict = {
        'train': os.path.join(data_path, 'train.list'),
        'val': os.path.join(data_path, 'val.list'),
        'test': os.path.join(data_path, 'test.list')
    }
    patch_size = (192, 192)

    if args.if_original:
        data_path = r"F:\pythonProject\Datasets\TeethData\2D\Graduate\01\data_list"
        img_path = r'F:\pythonProject\Datasets\TeethData\2D\Graduate\01\image'
        lab_path = r'F:\pythonProject\Datasets\TeethData\2D\Graduate\01\label'
        data_dict = {
            'train': os.path.join(data_path, 'train.list'),
            'val': os.path.join(data_path, 'val.list'),
            'test': os.path.join(data_path, 'test.list')
        }
        patch_size = (448, 448)

    # make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code')

    writer = SummaryWriter(snapshot_path + '/log/run%d' % args.max_iterations)

    logger.info(str(args))

    st_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 定义网络
    # net = Unet(in_ch=3, out_ch=num_classes)
    # net = net.to(device)

    # # smp 的 Unet
    # 原始Unet
    # net = smp.Unet(encoder_name='vgg11', encoder_weights='imagenet', in_channels=3, classes=num_classes)

    """
    ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 
    'resnext50_32x4d', 'resnext101_32x4d', 'resnext101_32x8d', 
    'resnext101_32x16d', 'resnext101_32x32d', 'resnext101_32x48d', 
    'dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn107', 'dpn131', 
    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 
    'senet154', 'se_resnet50', 'se_resnet101', 'se_resnet152', 
    'se_resnext50_32x4d', 'se_resnext101_32x4d', 
    'densenet121', 'densenet169', 'densenet201', 'densenet161', 
    'inceptionresnetv2', 'inceptionv4', 
    'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 
    'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7', 
    'mobilenet_v2', 'xception', 
    'timm-efficientnet-b0', 'timm-efficientnet-b1', 'timm-efficientnet-b2', 'timm-efficientnet-b3', 
    'timm-efficientnet-b4', 'timm-efficientnet-b5', 'timm-efficientnet-b6', 'timm-efficientnet-b7', 
    'timm-efficientnet-b8', 'timm-efficientnet-l2', 
    'timm-tf_efficientnet_lite0', 'timm-tf_efficientnet_lite1', 'timm-tf_efficientnet_lite2', 
    'timm-tf_efficientnet_lite3', 'timm-tf_efficientnet_lite4', 
    'timm-resnest14d', 'timm-resnest26d', 'timm-resnest50d', 'timm-resnest101e', 'timm-resnest200e', 
    'timm-resnest269e', 'timm-resnest50d_4s2x40d', 'timm-resnest50d_1s4x24d', 'timm-res2net50_26w_4s', 
    'timm-res2net101_26w_4s', 'timm-res2net50_26w_6s', 'timm-res2net50_26w_8s', 'timm-res2net50_48w_2s', 
    'timm-res2net50_14w_8s', 'timm-res2next50', 'timm-regnetx_002', 'timm-regnetx_004', 'timm-regnetx_006', 
    'timm-regnetx_008', 'timm-regnetx_016', 'timm-regnetx_032', 'timm-regnetx_040', 'timm-regnetx_064', 
    'timm-regnetx_080', 'timm-regnetx_120', 'timm-regnetx_160', 'timm-regnetx_320', 'timm-regnety_002', 
    'timm-regnety_004', 'timm-regnety_006', 'timm-regnety_008', 'timm-regnety_016', 'timm-regnety_032', 
    'timm-regnety_040', 'timm-regnety_064', 'timm-regnety_080', 'timm-regnety_120', 'timm-regnety_160', 
    'timm-regnety_320', 'timm-skresnet18', 'timm-skresnet34', 'timm-skresnext50_32x4d',
    'timm-mobilenetv3_large_075', 'timm-mobilenetv3_large_100', 'timm-mobilenetv3_large_minimal_100', 
    'timm-mobilenetv3_small_075', 'timm-mobilenetv3_small_100', 'timm-mobilenetv3_small_minimal_100', '
    timm-gernet_s', 'timm-gernet_m', 'timm-gernet_l', 
    'mit_b0', 'mit_b1', 'mit_b2', 'mit_b3', 'mit_b4', 'mit_b5', 
    'mobileone_s0', 'mobileone_s1', 'mobileone_s2', 'mobileone_s3', 'mobileone_s4']"
    
    """

    # res 下采样结构的Unet、可选参数：resnet18、resnet34、resnet50、resnet101、resnet152
    net = smp.Unet(encoder_name='resnet18', encoder_weights='imagenet', in_channels=3, classes=num_classes)

    # resNeXt 下采样结构的Unet、可选参数：resnext50_32x4d、resnext50_32x4d、resnext101_32x4d、resnext101_32x4d
    # net = smp.Unet(encoder_name='resnext50_32x4d', encoder_weights='imagenet', in_channels=3, classes=num_classes)

    # # monai的 Unet
    # net = AttentionUnet(
    #     spatial_dims=2,
    #     in_channels=3,
    #     out_channels=num_classes,
    #     channels=(16, 32, 64, 128),
    #     strides=()
    # )

    net.to(device)

    if args.use_state_model:
        net.load_state_dict(torch.load(args.state_model_path))

    train_transform = transforms.Compose([
        CropLeftMiddle(size=patch_size),  # 向左截取PatchSize的图片
        RandomRotate(p=0.5, rotateMax=10),  # 随机旋转
        RandomColorJitter(p=0.5, max_ratio=0.5),  # 随机增加亮度和对比度
        RandomVerticalFlip(p=0.5),  # 随机水平或垂直旋转
    ])
    val_transform = transforms.Compose([
        CropLeftMiddle(size=patch_size),  # 向左截取PatchSize的图片
        # RandomRotate(),
        # RandomColorJitter(),
        # RandomVerticalFlip(),
    ])
    mean_ = [0.485, 0.456, 0.406]
    std_ = [0.229, 0.224, 0.225]
    image_transform = transforms.Compose([
        transforms.ToTensor(),  # 原图转Tensor格式
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.Normalize(mean=mean_, std=std_)  # 再归一化
    ])

    label_transform = transforms.Compose([
        # LabelToBinary(),
        LabelToTensor(),  # 标签转Tensor格式
    ])

    # 获取数据
    db_train = Teeth(split='train',
                     img_path=img_path, lab_path=lab_path, data_path=data_dict,
                     transform=train_transform,
                     x_transform=image_transform,
                     y_transform=label_transform
                     )
    db_val = Teeth(split='val',
                   img_path=img_path, lab_path=lab_path, data_path=data_dict,
                   transform=val_transform,
                   x_transform=image_transform,
                   y_transform=label_transform
                   )

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    # 获取data_loader
    train_loader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(db_val, batch_size=batch_size, shuffle=True, num_workers=args.num_workers,
                            pin_memory=True, worker_init_fn=worker_init_fn)

    # 随机梯度下降优化器
    # optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    # Adam优化器
    optimizer = optim.Adam(net.parameters(), lr=base_lr, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1, last_epoch=-1)
    # 定义损失函数参数
    alpha = [0.01] + [1] * (num_classes - 1)  # focal Loss的权重
    alpha = torch.tensor(alpha).to(device)

    # Class Balanced所需要的有效样本数量
    sample_per_class = [99761,
                        7, 4, 7, 7, 8, 13, 12, 7,
                        5, 6, 7, 8, 10, 11, 7, 6,
                        5, 6, 7, 9, 10, 7, 8, 7,
                        5, 6, 8, 8, 5, 5, 2, 1]

    # 定义损失函数
    # ce_criterion = torch.nn.CrossEntropyLoss()
    use_criterion = torch.nn.CrossEntropyLoss(weight=alpha)
    loss_name = args.exp_which

    # # monai的
    if args.exp_which == 'FocalLoss':
        use_criterion = FocalLoss(to_onehot_y=True, weight=alpha)
    elif args.exp_which == 'DiceLoss':
        use_criterion = DiceLoss(to_onehot_y=True)
    elif args.exp_which == 'DiceCELoss':
        use_criterion = DiceCELoss(to_onehot_y=True, ce_weight=alpha)
    elif args.exp_which == 'DiceFocalLoss':
        use_criterion = DiceFocalLoss(to_onehot_y=True, focal_weight=alpha)

    # # 自己写的
    # fl_criterion = FocalLoss(alpha=alpha, gamma=2.0, num_class=num_classes, device=device)
    # dl_criterion = DiceLoss(num_class=num_classes, device=device)
    # cb_criterion = ClassBalancedLoss(gamma=2.0, beta=0.999, sample_per_class=sample_per_class,
    #                                  num_class=num_classes, device=device, loss_type='focal')

    iter_num = 0

    max_epoch = max_iterations // len(train_loader) + 1
    logger.info("共有{}个epoch, 每个epoch迭代 {} 次".format(max_epoch, len(train_loader)))
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        iter_num, base_lr = train_one_epoch(model=net, optimizer_=optimizer, snapshot_path=snapshot_path,
                                            criterion=use_criterion,
                                            loss_name=loss_name,
                                            writer_=writer, logger_=logger, max_iter=max_iterations,
                                            iter_num=iter_num, base_lr=base_lr, train_loader=train_loader)
        logger.info(f'训练第 {epoch_num} 轮，耗时：{time.time() - time1} s')
        if epoch_num % 5 == 0:
            time2 = time.time()
            evaluate_one_epoch(model=net, val_loader=val_loader,
                               criterion=use_criterion,
                               writer_=writer, logger_=logger, iter_num=iter_num, loss_name=loss_name)

            logger.info(f'在第 {epoch_num} 轮验证，耗时：{time.time() - time2} s')
        if iter_num > max_iterations:
            break

    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(max_iterations + 1) + '.pth')
    torch.save(net.state_dict(), save_mode_path)
    # logging.info("save model to {}".format(save_mode_path))
    logger.info(f'最后的模型保存在：{save_mode_path}')
    writer.close()
    logger.info(f'训练完毕。耗时：{time.time() - st_time}')


def calculate_dice_acc(score, target, logger_):
    # 转one hot label
    (b, c, h, w) = score.shape
    one_hot_label_batch = torch.zeros(b, c, h, w).cuda().scatter_(1, target.view(b, 1, h, w), 1)
    one_hot_label_batch.cuda()

    # logger_.info(f'out.shape:{outputs_soft.shape}, lab:{one_hot_label_batch.shape}')
    # 计算 dice 准确值
    dice_acc, dice_acc_no_zero = dice_multi(score, one_hot_label_batch)
    b_0_prediction = np.unique(np.argmax(score.cpu().detach().numpy()[0, :, :, :], axis=0))
    logger_.info(f'Dice准确率: {dice_acc:.6f}, 除0外的：{dice_acc_no_zero:.6f}，预测标签：{b_0_prediction}')
    del one_hot_label_batch, b_0_prediction
    return dice_acc, dice_acc_no_zero


# @logger.catch()
def train_one_epoch(model, optimizer_, train_loader, max_iter, criterion, loss_name,
                    writer_, logger_, iter_num, base_lr, snapshot_path):
    model.train()
    for i_batch, sampled_batch in enumerate(train_loader):
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
        outputs = model(volume_batch)

        outputs_soft = F.softmax(outputs, dim=1)
        label_batch = label_batch.unsqueeze(1)  # 从[N, H, W] -> [N, 1, H, W]
        # logger_.info(f'out.shape:{outputs.shape}, out_:{outputs_soft.shape}, lab:{label_batch.shape}')

        loss = criterion(outputs_soft, label_batch)

        dice_, dice_0 = calculate_dice_acc(outputs_soft, label_batch, logger_)

        logger_.info(f'iter:{iter_num:5}, {loss_name}:{loss:.5f}')
        writer_.add_scalar('loss/' + loss_name, loss, iter_num)
        writer_.add_scalar('acc/with_0', dice_, iter_num)
        writer_.add_scalar('acc/without_0', dice_0, iter_num)

        optimizer_.zero_grad()
        loss.backward()
        optimizer_.step()

        iter_num += 1

        # change lr
        if iter_num % 5000 == 0:
            base_lr = base_lr * 0.1 ** (iter_num // 2500)
            for param_group in optimizer_.param_groups:
                param_group['lr'] = base_lr

        writer_.add_scalar('learning_rate/lr_with_iter_num', base_lr, iter_num)
        if iter_num % 1000 == 0:
            save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            logger.info(f'模型保存在：{save_mode_path}')

        if iter_num > max_iter:
            break

    return iter_num, base_lr


# @logger.catch()
def evaluate_one_epoch(model, val_loader, iter_num, criterion, loss_name,
                       writer_, logger_):
    model.eval()  # 模型开始验证
    test_loss = 0  # 记录每个iter的损失
    iter_test = 0  # 是第几个iter
    dice_all_ = 0
    dice_all_0 = 0
    for i_batch, sampled_batch in enumerate(val_loader):
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
        with torch.no_grad():
            outputs = model(volume_batch)

        outputs_soft = F.softmax(outputs, dim=1)
        label_batch = label_batch.unsqueeze(1)  # 从[N, H, W] -> [N, 1, H, W]
        loss = criterion(outputs_soft, label_batch)
        logger_.info(f'train_iter:{iter_num:5}, val_iter:{i_batch:5}, {loss_name}:{loss:.5f}')

        dice_, dice_0 = calculate_dice_acc(outputs_soft, label_batch, logger_)

        dice_all_ += dice_
        dice_all_0 += dice_0
        test_loss = test_loss + loss
        iter_test = iter_test + 1
    writer_.add_scalar('loss_val/val_loss', test_loss / iter_test, iter_num)
    writer_.add_scalar('loss_val/with_0', dice_all_ / iter_test, iter_num)
    writer_.add_scalar('loss_val/without_0', dice_all_0 / iter_test, iter_num)
    model.train()
    del volume_batch, label_batch, outputs_soft, loss


if __name__ == "__main__":
    main()
