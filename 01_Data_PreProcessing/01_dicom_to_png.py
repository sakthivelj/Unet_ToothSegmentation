#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : 59-Lmq
# @Time     : 2023/3/30 15:37
# @File     : 01_dicom_to_png.py
# @Project  : ToothSegmentation_Unet

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import pandas as pd
from loguru import logger
import SimpleITK as sitk


def makedir(dir_path: str):
    """
    创建文件夹
    :param dir_path: 需要创建的文件夹路径
    :return: None
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logger.debug(f'文件夹：{dir_path}不存在，已创建完成')


def read_dicom(dcm_path: str):
    """
    读取DCM图像数据
    :param dcm_path: 输入的DCM文件夹路径
    :return: 返回翻转后的DCM图像array
    """
    reader = sitk.ImageSeriesReader()
    dcm_names = reader.GetGDCMSeriesFileNames(dcm_path)
    reader.SetFileNames(dcm_names)
    dicom = reader.Execute()
    dicom_array = sitk.GetArrayFromImage(dicom)  # 读取出来的shape:(z, y, x)
    dicom_array = dicom_array.transpose(2, 1, 0)  # 翻转后变为：(x, y, z)

    logger.info(f'DCM名为：{dcm_path} 所含DCM序列尺寸为：{dicom_array.shape}')
    return dicom_array
    # pass


def normalise_img(img):
    smooth = 1e-8
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + smooth)
    return img


def dicom2png():
    img_save_path = r'F:\pythonProject\Datasets\TeethData\2D\Graduate\image'
    img_ori_save_path = r'F:\pythonProject\Datasets\TeethData\2D\Graduate\image_ori'
    makedir(img_ori_save_path)
    makedir(img_save_path)

    dicom_path = r'F:\pythonProject\Datasets\TeethData\3D\OriginalData\CBCT-data\CT'
    st_time = time.time()
    dicom_folder_name = os.listdir(dicom_path)
    for folder_idx, folder_name in enumerate(dicom_folder_name):
        idx_time = time.time()
        logger.info(f'第 {folder_idx} 个数据，名为：{folder_name}')
        dicom_folder_path = os.path.join(dicom_path, folder_name)
        dicom_nd_array = read_dicom(dicom_folder_path)  # (x,y,z)

        plt.ion()
        for z in range(dicom_nd_array.shape[-1]):
            img = dicom_nd_array[:, :, z]
            img_pil_ori = Image.fromarray(img)

            img = normalise_img(img)
            img = np.uint(img * 256)
            img_pil = Image.fromarray(img)

            img_temp_path = os.path.join(img_save_path, folder_name + '_' + str(z).zfill(3) + '.png')
            img_ori_temp_path = os.path.join(img_ori_save_path, folder_name + '_' + str(z).zfill(3) + '.png')
            img_pil.convert('RGB').save(img_temp_path)
            img_pil_ori.convert('RGB').save(img_ori_temp_path)

            plt.subplot(121)
            title_name = folder_name + '_Z : ' + str(z).zfill(3)
            plt.title(title_name)
            plt.imshow(img_pil)

            plt.subplot(122)
            title_name_ori = folder_name + '_Original_Z : ' + str(z).zfill(3)
            plt.title(title_name_ori)
            plt.imshow(img_pil_ori)

            plt.pause(0.01)
            plt.clf()
        plt.ioff()
        idx_ed_time = time.time()
        logger.info(f'名为：{folder_name}，转换为PNG完毕，共有：{dicom_nd_array.shape[-1]}张，耗时：{idx_ed_time - idx_time} s')
        # break

    logger.info(f'DCM数据转为PNG完毕，耗时：{time.time() - st_time} s')


def record_image_data():
    image_detail_path = r'./data_detail/'
    makedir(image_detail_path)

    dicom_path = r'F:\pythonProject\Datasets\TeethData\3D\OriginalData\CBCT-data\CT'
    st_time = time.time()
    dicom_folder_name = os.listdir(dicom_path)

    for folder_idx, folder_name in enumerate(dicom_folder_name):
        idx_time = time.time()
        logger.info(f'第 {folder_idx} 个数据，名为：{folder_name}')
        dicom_folder_path = os.path.join(dicom_path, folder_name)
        dicom_nd_array = read_dicom(dicom_folder_path)  # (x,y,z)
        one_folder_dict = {}
        folder_one_png_dict = {}
        for z in range(dicom_nd_array.shape[-1]):
            img = dicom_nd_array[:, :, z]

            img = normalise_img(img)
            img = np.uint(img * 256)

            img_name = folder_name + '_' + str(z).zfill(3) + '.png'
            folder_one_png_dict.setdefault(img_name, (img.min(), img.max()))
            logger.info(f'{img_name}：min:{img.min()}, max:{img.max()}')

        one_folder_dict.setdefault(folder_name, folder_one_png_dict)

        one_folder_dict_pd = pd.DataFrame(one_folder_dict)
        folder_detail_name = os.path.join(image_detail_path, 'detail_' + folder_name + '.xlsx')
        one_folder_dict_pd.to_excel(folder_detail_name)

        idx_ed_time = time.time()
        logger.info(f'名为：{folder_name}的原图记录完毕，保存到：{folder_detail_name}，耗时：{idx_ed_time - idx_time} s')
        # break

    logger.info(f'DCM数据记录完毕，耗时：{time.time() - st_time} s')


if __name__ == '__main__':
    log_path = r'./log/01_dicom_to_png_'
    logger.add(log_path + '{time}.log', rotation='00:00')

    # dicom2png()
    record_image_data()
