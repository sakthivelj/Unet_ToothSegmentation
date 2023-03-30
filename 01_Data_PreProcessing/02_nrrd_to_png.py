#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : 59-Lmq
# @Time     : 2023/3/30 17:17
# @File     : 02_nrrd_to_png.py
# @Project  : ToothSegmentation_Unet
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import pandas as pd
from loguru import logger

import nrrd


def makedir(dir_path: str):
    """
    创建文件夹
    :param dir_path: 需要创建的文件夹路径
    :return: None
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logger.debug(f'文件夹：{dir_path}不存在，已创建完成')


def read_nrrd(nrrd_path):
    label_array, options = nrrd.read(nrrd_path, index_order='F')

    logger.info(f'名为：{nrrd_path} 所含序列尺寸为：{label_array.shape}')
    return label_array


def nrrd2png():
    label_save_path = r'F:\pythonProject\Datasets\TeethData\2D\Graduate\01\label'
    makedir(label_save_path)

    nrrd_path = r'F:\pythonProject\Datasets\TeethData\3D\FixData\RT_CBCT_data_17\nrrd_label\label_fixed'
    st_time = time.time()
    nrrd_name = os.listdir(nrrd_path)
    for idx, name in enumerate(nrrd_name):
        if idx <= 7:
            continue
        idx_time = time.time()
        logger.info(f'第 {idx} 个标签数据，名为：{name}')
        nrrd_file_path = os.path.join(nrrd_path, name)
        nrrd_nd_array = read_nrrd(nrrd_file_path)
        # fig, axs = plt.subplots()
        plt.ion()
        for z in range(nrrd_nd_array.shape[-1]):

            label = nrrd_nd_array[:, :, z]
            label_pil = Image.fromarray(label)

            label_name = 'CBCT' + name.split('.')[0].split('l')[1] + '_' + str(z).zfill(3) + '.png'
            label_temp_path = os.path.join(label_save_path, label_name)
            label_pil.convert('RGB').save(label_temp_path)

            title_name_2 = name + '_heat_Z : ' + str(z).zfill(3)
            plt.title(title_name_2)
            plt.imshow(label, cmap='hot', vmin=0, vmax=33)
            # logger.debug(f'label.unique:{np.unique(label)}')
            plt.colorbar()

            plt.pause(0.001)
            plt.clf()
        plt.ioff()
        i_ed_time = time.time()
        logger.info(f'第 {idx} 个名为{name}，转换为PNG完毕，共有：{nrrd_nd_array.shape[-1]}张，耗时：{i_ed_time - idx_time} s')
        # break

    logger.info(f'NRRD标签数据转为PNG完毕，耗时：{time.time() - st_time} s')


def record_label_data():
    label_detail_path = r'./data_detail/'
    makedir(label_detail_path)

    nrrd_path = r'F:\pythonProject\Datasets\TeethData\3D\FixData\RT_CBCT_data_17\nrrd_label\label_fixed'
    st_time = time.time()
    nrrd_name = os.listdir(nrrd_path)

    nrrd_dict = {}
    for idx, name in enumerate(nrrd_name):
        idx_time = time.time()
        logger.info(f'第 {idx} 个标签数据，名为：{name}')
        nrrd_file_path = os.path.join(nrrd_path, name)
        nrrd_nd_array = read_nrrd(nrrd_file_path)

        one_nrrd_dict = {}
        for c in range(33):
            c_sum = np.sum(nrrd_nd_array == c)
            one_nrrd_dict.setdefault(str(c), c_sum)
            logger.info(f'{name}.sum: {c}: {c_sum}')
        nrrd_dict.setdefault(name, one_nrrd_dict)

        one_png_dict = {}
        for z in range(nrrd_nd_array.shape[-1]):
            label = nrrd_nd_array[:, :, z]

            label_name = name.split('.')[0] + '_' + str(z).zfill(3) + '.png'
            one_png_label_dict = {}

            for c in range(33):
                c_sum = np.sum(label == c)
                one_png_label_dict.setdefault(str(c), c_sum)
                logger.info(f'{label_name}.sum: {c}: {c_sum}')
            one_png_dict.setdefault(label_name, one_png_label_dict)
            # break

        nrrd_detail_name = os.path.join(label_detail_path, 'detail_' + name.split('.')[0] + '.xlsx')
        one_png_dict_pd = pd.DataFrame(one_png_dict)
        one_png_dict_pd.T.to_excel(nrrd_detail_name)
        idx_ed_time = time.time()
        logger.info(f'名为：{name}，记录完毕，保存到：{nrrd_detail_name}，耗时：{idx_ed_time - idx_time} s')
        # break

    nrrd_dict_pd = pd.DataFrame(nrrd_dict)
    data_detail_name = os.path.join(label_detail_path, 'label_detail.xlsx')
    nrrd_dict_pd.to_excel(data_detail_name)

    logger.info(f'NRRD标签数据记录完毕，保存在：{data_detail_name}, 耗时：{time.time() - st_time} s')


def heat_test():
    import numpy as np
    import matplotlib.pyplot as plt

    # 生成0-33的灰度图
    gray_img = np.random.randint(low=0, high=32, size=(667, 667))

    # 将灰度图转换为热图
    heatmap = plt.imshow(gray_img, cmap='hot', vmin=0, vmax=32)
    plt.colorbar()  # 显示颜色条

    # 显示图像
    plt.show()


if __name__ == '__main__':
    log_path = r'./log/02_nrrd_to_png_'
    logger.add(log_path + '{time}.log', rotation='00:00')

    nrrd2png()
    # heat_test()
    # record_label_data()
