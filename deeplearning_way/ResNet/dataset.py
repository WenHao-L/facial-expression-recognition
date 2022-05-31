import os

import cv2
import numpy as np
import pandas as pd
from torchvision.transforms import transforms
from torch.utils.data import Dataset
import torch


EMOTION_DICT = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}
"""
预处理
"""
def tencrop(crops):
    return torch.stack([transforms.ToTensor()(crop) for crop in crops])
def normal(tensors):
    mu, st = 0.5086, 0.2481
    return torch.stack([transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors])

def erasing(tensors):
    return torch.stack([transforms.RandomErasing()(t) for t in tensors])


class FER2013(Dataset):
    def __init__(self, stage, configs,svm = False, tta=False, tta_size=48):
        self._stage = stage    #训练 测试 验证
        self._configs = configs
        self._channels = configs["in_channels"]
        self._tta = tta
        self._tta_size = tta_size
        self._svm = svm

        self._image_size = (configs["image_size"], configs["image_size"])

        self._data = pd.read_csv(configs["data_path"])

        self._data = self._data[self._data['Usage'] == self._stage] #提取数据大小

        self._pixels = self._data["pixels"].tolist()

        self.data_size = len(self._pixels)

        # self._label = pd.read_csv('fer2013new1.csv')[["neutral","happiness","surprise","sadness","anger","disgust","fear"]]

        # self._label_p = np.array(self._label) #np!

        self._emotions = pd.get_dummies(self._data["emotion"])  #热键编码

        mu, st = 0.5086, 0.2481

        self._train_transform = transforms.Compose([

                    transforms.ToPILImage(),
                    transforms.RandomApply([transforms.ColorJitter(
                        brightness=0.5, contrast=0.5, saturation=0.5)], p=0.5),  #随机更改图像的亮度、对比度、饱和度和色调
                    transforms.RandomApply([transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),  #图像保持中心不变的随机仿射变换

                    # 输入pil,输出tensor
                    # transforms.TenCrop(200),            #将给定的图像裁剪为四个角，并裁剪为中央。40：size 输入必须为PIL
                    # transforms.Lambda(tencrop),   # 不能直接用lambda，否则会报错，原因：win11上面不支持 No, it is not supported on Windows. The reason is that multiprocessing lib doesn’t have it implemented on Windows. There are some alternatives like dill that can pickle more objects.
                    transforms.ToTensor(),
                    # transforms.Lambda(normal),
                    # transforms.Lambda(erasing), #随机选择火炬张量图像中的矩形区域并擦除其像素
                    transforms.Normalize(mean=(mu,), std=(st,)),
                    transforms.RandomErasing(),

                    #输入tensor,输出tensor
                ])



    def is_tta(self):  #测试使用
        return self._tta == True

    def __len__(self):
        return len(self._pixels)

    def __getitem__(self, idx):  #什么时候使用？Dataloader
        ###    把数据从str进行预处理，并且放大况且复制成3通道
        pixels = self._pixels[idx]   #选取一张图片
        pixels = list(map(int, pixels.split(" "))) #str以空格分开生成数组
        image = np.asarray(pixels).reshape(48, 48)
        image = image.astype(np.uint8)   #int32 -> uint8
        image = cv2.resize(image, self._image_size)  #放大
        image = np.dstack([image] * 3)

        #数据增强,对于训练和验证集用一样的预处理方法
        image = self._train_transform(image)


        target = self._emotions.iloc[idx].idxmax()

        return image, target

class FER2013_test(Dataset):
    def __init__(self, stage, configs,svm = False, tta=False, tta_size=48):
        self._stage = stage    #训练 测试 验证
        self._configs = configs
        self._channels = configs["in_channels"]
        self._tta = tta
        self._tta_size = tta_size
        self._svm = svm

        self._image_size = (configs["image_size"], configs["image_size"])

        self._data = pd.read_csv(configs["data_path"])

        self._data = self._data[self._data['Usage'] == self._stage]#提取数据大小

        self._pixels = self._data["pixels"].tolist()

        self.data_size = len(self._pixels)

        # self._label = pd.read_csv('fer2013new1.csv')[["neutral","happiness","surprise","sadness","anger","disgust","fear"]]

        # self._label_p = np.array(self._label) #np!

        self._emotions = pd.get_dummies(self._data["emotion"])  #热键编码

        mu, st = 0.5086, 0.2481

        # tta
        self._test_transform = transforms.Compose([

            transforms.ToPILImage(),
            # transforms.ToTensor(), #转换数据格式，把数据转换为tensfroms格式。只有转换为tensfroms格式才能进行后面的处理。
            # transforms.RandomApply([transforms.ColorJitter(
            #     brightness=0.5, contrast=0.5, saturation=0.5)], p=0.5),  #随机更改图像的亮度、对比度、饱和度和色调
            # transforms.RandomApply([transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),  #图像保持中心不变的随机仿射变换
            transforms.FiveCrop(224),            #将给定的图像裁剪为四个角，并裁剪为中央。40：size 输入必须为PIL
            transforms.Lambda(tencrop), # 10 3 224 224
            # transforms.Lambda(normal),
            # transforms.Lambda(erasing), #随机选择火炬张量图像中的矩形区域并擦除其像素

            # transforms.ToTensor(), #转换为tensor格式，这个格式可以直接输入进神经网络了。
        ])



    def is_tta(self):  #测试使用
        return self._tta == True

    def __len__(self):
        return len(self._pixels)

    def __getitem__(self, idx):  #什么时候使用？Dataloader
        ###    把数据从str进行预处理，并且放大况且复制成3通道
        pixels = self._pixels[idx]   #选取一张图片
        pixels = list(map(int, pixels.split(" "))) #str以空格分开生成数组
        image = np.asarray(pixels).reshape(48, 48)
        image = image.astype(np.uint8)   #int32 -> uint8
        image = cv2.resize(image, self._image_size)  #放大
        image = np.dstack([image] * 3)

        image = self._test_transform(image)
        target = self._emotions.iloc[idx].idxmax()

        return image, target


class FER2013_test_without_tta(Dataset):
    def __init__(self, stage, configs,svm = False, tta=False, tta_size=48):
        self._stage = stage    #训练 测试 验证
        self._configs = configs
        self._channels = configs["in_channels"]
        self._tta = tta
        self._tta_size = tta_size
        self._svm = svm

        self._image_size = (configs["image_size"], configs["image_size"])

        self._data = pd.read_csv(configs["data_path"])

        self._data = self._data[self._data['Usage'] == self._stage]#提取数据大小

        self._pixels = self._data["pixels"].tolist()

        self.data_size = len(self._pixels)

        # self._label = pd.read_csv('fer2013new1.csv')[["neutral","happiness","surprise","sadness","anger","disgust","fear"]]

        # self._label_p = np.array(self._label) #np!

        self._emotions = pd.get_dummies(self._data["emotion"])  #热键编码

        mu, st = 0.5086, 0.2481


        self._test_transform = transforms.Compose([

            # transforms.ToPILImage(),
            transforms.ToTensor(), #转换数据格式，把数据转换为tensfroms格式。只有转换为tensfroms格式才能进行后面的处理。
            # transforms.TenCrop(224),            #将给定的图像裁剪为四个角，并裁剪为中央。40：size 输入必须为PIL


            transforms.Normalize(mean=(mu,), std=(st,)),
            transforms.RandomErasing(),
            # transforms.ToTensor(), #转换为tensor格式，这个格式可以直接输入进神经网络了。
        ])



    def is_tta(self):  #测试使用
        return self._tta == True

    def __len__(self):
        return len(self._pixels)

    def __getitem__(self, idx):  #什么时候使用？Dataloader
        ###    把数据从str进行预处理，并且放大况且复制成3通道
        pixels = self._pixels[idx]   #选取一张图片
        pixels = list(map(int, pixels.split(" "))) #str以空格分开生成数组
        image = np.asarray(pixels).reshape(48, 48)
        image = image.astype(np.uint8)   #int32 -> uint8
        image = cv2.resize(image, self._image_size)  #放大
        image = np.dstack([image] * 3)

        image = self._test_transform(image)
        target = self._emotions.iloc[idx].idxmax()

        return image, target