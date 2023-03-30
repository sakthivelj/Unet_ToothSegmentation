#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : 59-Lmq
# @Time     : 2023/3/30 20:23
# @File     : 03_data_split_train_val_test.py
# @Project  : ToothSegmentation_Unet

import os
import time

from loguru import logger
from sklearn.model_selection import train_test_split


def makedir(dir_path: str):
    """
    创建文件夹
    :param dir_path: 需要创建的文件夹路径
    :return: None
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logger.debug(f'文件夹：{dir_path}不存在，已创建完成')


def original_data_list():
    image_path = r'F:\pythonProject\Datasets\TeethData\2D\Graduate\01\image'
    label_path = r'F:\pythonProject\Datasets\TeethData\2D\Graduate\01\label'
    data_list_path = r'F:\pythonProject\Datasets\TeethData\2D\Graduate\01\data_list'
    makedir(data_list_path)

    image_name = os.listdir(image_path)
    label_name = os.listdir(label_path)

    image_full_name, label_full_name = [], []
    for i, l in zip(image_name, label_name):
        image_full_name.append(os.path.join(image_path, i))
        label_full_name.append(os.path.join(label_path, l))

    image_list_name = os.path.join(data_list_path, 'image.list')
    with open(image_list_name, 'w') as f:
        f.write('\n'.join(image_full_name))

    label_list_name = os.path.join(data_list_path, 'label.list')
    with open(label_list_name, 'w') as f:
        f.write('\n'.join(label_full_name))


def split_original():
    """
    两个条件
    1、原图中CBCT切片不满500张的数据丢弃（无法切割到512，512），具体数据名为：CBCT2207156、CBCT2208186、CBCT2208200
    2、CBCT切片中，超过570的数据丢弃（全是0）
    :return:
    """
    st_time = time.time()
    dicom_path = r'F:\pythonProject\Datasets\TeethData\3D\OriginalData\CBCT-data\CT'
    image_path = r'F:\pythonProject\Datasets\TeethData\2D\Graduate\01\image'
    data_list_path = r'F:\pythonProject\Datasets\TeethData\2D\Graduate\01\data_list'
    makedir(data_list_path)

    CBCT_data_name = os.listdir(dicom_path)
    image_name = os.listdir(image_path)

    drop_name = ['CBCT2208200', 'CBCT2208186', 'CBCT2207156']
    drop_line = 570

    CBCT_data_name_drop = []
    for i in CBCT_data_name:
        if i not in drop_name:
            CBCT_data_name_drop.append(i)
    logger.info(f'CBCT_data_name_drop.len:{len(CBCT_data_name_drop)}')

    train_name, val_test_name = train_test_split(CBCT_data_name_drop, test_size=0.2, random_state=3407, shuffle=True)
    val_name, test_name = train_test_split(val_test_name, test_size=0.5, random_state=3407, shuffle=True)

    logger.info(f'train_name:len:{len(train_name)}\t:{train_name}')
    logger.info(f'val_name:len:{len(val_name)}\t:{val_name}')
    logger.info(f'test_name:len:{len(test_name)}\t:{test_name}')

    train_list, val_list, test_list = [], [], []
    for i, img_name in enumerate(image_name):
        img_name_prefix = img_name.split('.')[0].split('_')[0]
        slice_num = int(img_name.split('.')[0].split('_')[1])

        if slice_num > drop_line:
            continue

        if img_name_prefix in train_name:
            train_list.append(img_name)
        elif img_name_prefix in val_name:
            val_list.append(img_name)
        elif img_name_prefix in test_name:
            test_list.append(img_name)

    train_list_name = os.path.join(data_list_path, 'train.list')
    with open(train_list_name, 'w') as f:
        f.write('\n'.join(train_list))

    val_list_name = os.path.join(data_list_path, 'val.list')
    with open(val_list_name, 'w') as f:
        f.write('\n'.join(val_list))

    test_list_name = os.path.join(data_list_path, 'test.list')
    with open(test_list_name, 'w') as f:
        f.write('\n'.join(test_list))

    logger.info(f'png train_list:len:{len(train_list)}')
    logger.info(f'png val_list:len:{len(val_list)}')
    logger.info(f'png test_list:len:{len(test_list)}')

    logger.info(f'原图划分数据集完毕，train:val:test 为 8：1：1，对应的数据量如上，总耗时：{time.time() - st_time} s')


if __name__ == '__main__':

    log_path = r'./log/03_data_split_train_val_test_'
    logger.add(log_path + '{time}.log', rotation='00:00')

    # original_data_list()
    split_original()
