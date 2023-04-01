import os
import torch
import numpy as np

from torch.utils.data import Dataset
import cv2
import PIL.Image as Image
from PIL import ImageEnhance, ImageOps
import random
import matplotlib.pyplot as plt


class Teeth(Dataset):
    """牙齿二维分割"""

    def __init__(self, split='train',
                 img_path=None, lab_path=None,
                 num=None, data_path=None,
                 x_transform=None, y_transform=None, transform=None
                 ):

        self.transform = transform
        self.x_transform = x_transform
        self.y_transform = y_transform

        self.sample_list = []
        self.data_path = data_path
        self.img_path = img_path
        self.lab_path = lab_path

        if split == 'train':
            with open(self.data_path['train'], 'r') as f:
                self.sample_list = f.readlines()
        elif split == 'val':
            with open(self.data_path['val'], 'r') as f:
                self.sample_list = f.readlines()
        elif split == 'test':
            with open(self.data_path['test'], 'r') as f:
                self.sample_list = f.readlines()

        self.sample_list = [item.replace('\n', '') for item in self.sample_list]
        if num is not None:
            self.sample_list = self.sample_list[:num]
        print(f"{split} total {len(self.sample_list)} samples")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        image_name = self.sample_list[idx]
        img_name = os.path.join(self.img_path, image_name)
        lab_name = os.path.join(self.lab_path, image_name)

        image = Image.open(img_name)
        label = Image.open(lab_name)
        label = label.convert("L")  # 转灰度

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        if self.x_transform:
            image = sample['image']
            image = self.x_transform(image)
        if self.y_transform:
            label = sample['label']
            label = self.y_transform(label)
        sample = {'image': image, 'label': label}
        return sample


class CropLeftMiddle(object):
    def __init__(self, size: int or list, left_margin_offset: float = 0.1):
        if isinstance(size, int):
            crop_size = (size, size)
        elif isinstance(size, tuple):
            crop_size = size
        else:
            print("Error: CropLeftMiddle: Crop size is illegal")
            raise ValueError
        self.crop_size = crop_size
        self.lmo = left_margin_offset

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        img_size = image.size

        Vertical_crop_Middle = img_size[0] / 2
        y1, y2 = int(Vertical_crop_Middle - self.crop_size[1] / 2), int(Vertical_crop_Middle + self.crop_size[1] / 2)
        crop_window = (0 + int(img_size[0] * self.lmo), y1, self.crop_size[0] + int(img_size[0] * self.lmo), y2)
        return {'image': image.crop(crop_window), 'label': label.crop(crop_window)}


class RandomColorJitter(object):
    def __init__(self, p: float = 0.5, max_ratio: float = 0.2):
        self.p = p
        self.max_ratio = max_ratio

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        random_p = random.random()
        if random_p <= self.p:
            random_p = random.random()
            ratio = random_p * self.max_ratio * 2 - self.max_ratio + 1
            bright_enhancer = ImageEnhance.Brightness(image)
            # labelmap_bright_enhancer = ImageEnhance.Brightness(labelmap)
            bright_img = bright_enhancer.enhance(ratio)
            contrast_enhancer = ImageEnhance.Contrast(bright_img)
            image = contrast_enhancer.enhance(ratio)
        return {'image': image, 'label': label}


class RandomRotate(object):
    def __init__(self, p: float = 0.5, rotateMax: float = 10):
        self.p = p
        self.rotate_Max = rotateMax

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        random_p = random.random()
        if random_p <= self.p:
            random_p = random.random()
            angle = random_p * 2 * self.rotate_Max - self.rotate_Max
            image = image.rotate(angle)
            label = label.rotate(angle)
        return {'image': image, 'label': label}


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, int):
            crop_size = (size, size)
        elif isinstance(size, tuple):
            crop_size = size
        else:
            print("Error: CropLeftMiddle: Crop size is illegal")
            raise ValueError
        self.crop_size = crop_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        (w, h, d) = image.size

        # 左上角坐标
        w_1 = int(round((w - self.crop_size[0]) / 2.))
        h_1 = int(round((h - self.crop_size[1]) / 2.))

        # 右下角坐标
        w_2 = w_1 + self.crop_size[0]
        h_2 = h_1 + self.crop_size[1]

        crop_window = (w_1, h_1, w_2, h_2)
        return {'image': image.crop(crop_window), 'label': label.crop(crop_window)}


class RandomCropResize(object):
    def __init__(self,
                 p: float = 0.5,
                 min_ratio: float = 0.6,
                 max_ratio: int or float = 1):
        self.p = p
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        img_size = image.size
        random_p = random.random()
        if random_p <= self.p:
            random_p = random.random()
            scale = self.min_ratio + random_p * (self.max_ratio - self.min_ratio)

            new_h = int(img_size[1] * scale)
            new_w = int(img_size[0] * scale)
            y = np.random.randint(0, img_size[1] - new_h)
            x = np.random.randint(0, img_size[0] - new_w)

            image = image.crop((x, y, x + new_w, y + new_h))
            label = label.crop((x, y, x + new_w, y + new_h))

        return {'image': image, 'label': label}


class RandomVerticalFlip(object):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        random_p = random.random()
        if random_p <= self.p:
            image = ImageOps.flip(image)
            label = ImageOps.flip(label)
        return {'image': image, 'label': label}


class LabelToTensor(object):
    def __call__(self, label):
        label = np.array(label).astype(np.float32)
        return torch.from_numpy(label).long()


class LabelToBinary(object):
    def __call__(self, label):
        label = label.point(lambda x: 1 if x > 0 else x)
        return label


class LabelToMulti(object):
    def __init__(self, label_dict):
        self.label_dict = label_dict

    def __call__(self, label):
        for original_label, new_label in self.label_dict:
            label = label.point(lambda x: new_label if x == original_label else x)
        return label


def plot_data_loader(data_loader, std, mean):
    plt.ion()  # 开启动态模式
    for batch_idx, sample in enumerate(data_loader):
        img = sample['image']  # [N, C, H, W]
        lab = sample['label']  # [N, H, W]

        print(f'img.shape:{img.shape}, lab.shape:{lab.shape}')
        print(f'img.shape:{img.shape}, lab.unique:{np.unique(lab)}')
        # img_ = img.squeeze(0)
        img_ = img[0, :, :, :]  # 取第一个batch的图片
        img_ = img_.permute(1, 2, 0)  # 从 [C, H, W] -> [H, W, C]
        img_ = img_.cpu().numpy()  # 转为numpy数组

        # 反归一化，注意上面原图做了个transform.Normalise(mean, std)
        img_ = img_ * std + mean
        # img_ = img_.astype(np.float32) / 255.0

        # lab_ = lab.squeeze(0)
        lab_ = lab[0, :, :]  # 取第一个batch的标签
        lab_ = lab_.cpu().numpy().astype(np.uint8)  # 转为无符号8位的数据

        # # 注释是用cv2实现热图显示的代码，不好用，但能用
        # lab_heat = cv2.applyColorMap((lab_ * 255 / 33).astype(np.uint8), cv2.COLORMAP_PINK)
        #
        # img_show_all = np.hstack([img_, lab_heat])
        # cv2.imshow('img_show', img_show_all)
        # cv2.waitKey(0)

        # 下面是用plt显示原图和标签热图的代码，可用
        plt.subplot(121)  # 定义画布左边起第一个图
        plt.imshow(img_)  # 显示原图

        plt.subplot(122)  # 定义画布左边起第二个图
        plt.imshow(lab_, cmap='hot', vmin=0, vmax=32)  # 显示标签，cmap='hot'表示热图，vmin是标签的下值，vmax是标签的上值
        # plt.imshow(lab_, cmap='hot', vmin=0, vmax=1)
        plt.pause(1)  # 画面显示多少秒
        if batch_idx > 1:
            break
    plt.ioff()  # 关闭动态模式


def test_data_loader():
    import cv2
    from torch.utils.data import DataLoader
    from torchvision import transforms
    data_path = r"F:\pythonProject\Datasets\TeethData\2D\Graduate\01\data_list"
    img_path = r'F:\pythonProject\Datasets\TeethData\2D\Graduate\01\image'
    lab_path = r'F:\pythonProject\Datasets\TeethData\2D\Graduate\01\label'
    data_dict = {
        'train': os.path.join(data_path, 'train.list'),
        'val': os.path.join(data_path, 'val.list'),
        'test': os.path.join(data_path, 'test.list')
    }

    batch_size = 2
    patch_size = (448, 448)
    # patch_size = (512, 512)
    train_transform = transforms.Compose([
        CropLeftMiddle(size=patch_size),
        RandomRotate(),
        RandomColorJitter(),
        RandomVerticalFlip(),
    ])
    val_transform = transforms.Compose([
        CropLeftMiddle(size=patch_size),
        # RandomRotate(),
        # RandomColorJitter(),
        # RandomVerticalFlip(),
    ])
    test_transform = transforms.Compose([
        # CropLeftMiddle(size=patch_size),
        # RandomRotate(),
        # RandomColorJitter(),
        # RandomVerticalFlip(),
    ])
    std_ = [0.229, 0.224, 0.225]
    mean_ = [0.485, 0.456, 0.406]
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.Normalize(mean=mean_, std=std_)
    ])

    label_transform = transforms.Compose([
        # LabelToBinary(),
        LabelToTensor(),
    ])

    train_teeth = Teeth(split='train', img_path=img_path, lab_path=lab_path, data_path=data_dict,
                     transform=train_transform,
                     x_transform=image_transform,
                     y_transform=label_transform
                     )

    val_teeth = Teeth(split='val', img_path=img_path, lab_path=lab_path, data_path=data_dict,
                      transform=val_transform,
                      x_transform=image_transform,
                      y_transform=label_transform
                      )

    test_teeth = Teeth(split='val', img_path=img_path, lab_path=lab_path, data_path=data_dict,
                       transform=test_transform,
                       x_transform=image_transform,
                       y_transform=label_transform
                       )

    train_loader = DataLoader(train_teeth, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f'len(train_loader):{len(train_loader)}')

    val_loader = DataLoader(val_teeth, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f'len(val_loader):{len(val_loader)}')

    test_loader = DataLoader(test_teeth, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f'len(test_loader):{len(test_loader)}')

    print(f'开始画 train_loader')
    plot_data_loader(train_loader, std=std_, mean=mean_)

    print(f'开始画 val_loader')
    plot_data_loader(val_loader, std=std_, mean=mean_)

    print(f'开始画 test_loader')
    plot_data_loader(test_loader, std=std_, mean=mean_)

    # plt.ion()  # 开启动态模式
    # for batch_idx, sample in enumerate(train_loader):
    #     img = sample['image']  # [N, C, H, W]
    #     lab = sample['label']  # [N, H, W]
    #
    #     print(f'img.shape:{img.shape}, lab.shape:{lab.shape}')
    #     print(f'img.shape:{img.shape}, lab.unique:{np.unique(lab)}')
    #     # img_ = img.squeeze(0)
    #     img_ = img[0, :, :, :]  # 取第一个batch的图片
    #     img_ = img_.permute(1, 2, 0)  # 从 [C, H, W] -> [H, W, C]
    #     img_ = img_.cpu().numpy()  # 转为numpy数组
    #
    #     # 反归一化，注意上面原图做了个transform.Normalise(mean, std)
    #     img_ = img_ * std_ + mean_
    #     # img_ = img_.astype(np.float32) / 255.0
    #
    #     # lab_ = lab.squeeze(0)
    #     lab_ = lab[0, :, :]  # 取第一个batch的标签
    #     lab_ = lab_.cpu().numpy().astype(np.uint8)  # 转为无符号8位的数据
    #
    #     # # 注释是用cv2实现热图显示的代码，不好用，但能用
    #     # lab_heat = cv2.applyColorMap((lab_ * 255 / 33).astype(np.uint8), cv2.COLORMAP_PINK)
    #     #
    #     # img_show_all = np.hstack([img_, lab_heat])
    #     # cv2.imshow('img_show', img_show_all)
    #     # cv2.waitKey(0)
    #
    #     # 下面是用plt显示原图和标签热图的代码，可用
    #     plt.subplot(121)  # 定义画布左边起第一个图
    #     plt.imshow(img_)  # 显示原图
    #
    #     plt.subplot(122)  # 定义画布左边起第二个图
    #     plt.imshow(lab_, cmap='hot', vmin=0, vmax=32)  # 显示标签，cmap='hot'表示热图，vmin是标签的下值，vmax是标签的上值
    #     # plt.imshow(lab_, cmap='hot', vmin=0, vmax=1)
    #     plt.pause(0.1)  # 画面显示多少秒
    #     # if batch_idx > 5:
    #     #     break
    # plt.ioff()  # 关闭动态模式


if __name__ == '__main__':
    test_data_loader()
