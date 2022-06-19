import numpy as np
import mindspore.dataset as ds
import pandas as pd
import cv2

np.random.seed(1)

class MyDataset:
    """自定义数据集类"""

    def __init__(self,stage):
        """自定义初始化操作"""
        self._stage = stage    #训练 测试 验证

        self._channels = 3


        self._image_size = (48,48)

        self._data = pd.read_csv("my.csv")

        self._data = self._data[self._data['Usage'] == self._stage] #提取数据大小

        self._pixels = self._data["pixels"].tolist()

        self.data_size = len(self._pixels)

        self._emotions = pd.get_dummies(self._data["emotion"])  #热键编码

    def __getitem__(self, index):
        """自定义随机访问函数"""
        pixels = self._pixels[index]   #选取一张图片
        pixels = list(map(int, pixels.split(" "))) #str以空格分开生成数组
        image = np.asarray(pixels).reshape(48, 48)
        image = image.astype(np.uint8)   #int32 -> uint8
        # image = cv2.resize(image, self._image_size)  #放大
        image = np.dstack([image] * 3)

        self.data[index] = image
        self.label[index] = self._emotions.iloc[index].idxmax()

        return self.data[index], self.label[index]

    def __len__(self):
        """自定义获取样本数据量函数"""
        return len(self._pixels)

# 实例化数据集类
dataset_generator = MyDataset("val_set")


print(dataset_generator)
# dataset = ds.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=False)

# # 迭代访问数据集
# for data in dataset.create_dict_iterator():
#     # data1 = data['data'].asnumpy()
#     # label1 = data['label'].asnumpy()
#     # print(f'data:[{data1[0]:7.5f}, {data1[1]:7.5f}], label:[{label1[0]:7.5f}]')
#     data1 = data['data']
#     label = data["label"]

# # 打印数据条数
# print("data size:", dataset.get_dataset_size())
