import numpy as np
import pandas as pd
import mindspore as ms
import mindspore.nn as nn
import mindspore.dataset as ds
from PIL import Image

import mindspore.dataset.vision.py_transforms as py_vision
from mindspore.dataset.transforms.py_transforms  import Compose
from mindspore.dataset.transforms import py_transforms
from mindspore import context
context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
# context.set_context(mode=context.GRAPH_MODE)
# context.set_context(device_target="Ascend")

from model import resnet18

from mindspore.train.callback import LossMonitor
from mindspore.train.callback import ModelCheckpoint, LossMonitor

# 定义回调类

loss_cb = LossMonitor()

## 2. 手动生成器
class DatasetGenerator:
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        return (image, label)

    def __len__(self):
        return len(self.images)

## 1. 数据处理
def prepare_data(data):
    image_array = np.zeros(shape=(len(data), 48, 48, 3))
    image_label = np.array(list(map(int, data['emotion'])))

    for i, row in enumerate(data.index):
        image = np.fromstring(data.loc[row, 'pixels'], dtype=int, sep=' ')
        image = np.reshape(image, (48, 48))
        # image = np.asarray(image).reshape(48,48)
        # image = image.astype(np.uint8)
        image = np.dstack([image] * 3)
        image = Image.fromarray(np.uint8(image))
        image_array[i] = image     # 1. array  2. np.ndarray

    return image_array, image_label


def ms_load_dataset():
    path = './datasets/fer2013.csv'
    fer2013 = pd.read_csv(path)
    train_img, train_label = prepare_data(fer2013[fer2013['Usage'] == 'PublicTest'])

    train_transforms_list = Compose([
           #  py_vision.ToTensor(),
            py_vision.ToPIL(),
            py_vision.RandomResizedCrop(size=(48, 48), scale=(0.8, 1.2)),
            py_transforms.RandomApply([py_vision.RandomAffine(degrees=0, translate=(0.2, 0.2))], prob=0.5),
            py_vision.RandomHorizontalFlip(prob=0.5),
            py_vision.ToTensor(),


    ])

    train_dataset_generator = DatasetGenerator(train_img, train_label)

    ## 3. 创建dataset
    train_dataset = ds.GeneratorDataset(train_dataset_generator, ["data", "label"], shuffle=True, num_parallel_workers=1) #生成dataset

    ## 4. 数据增强
    train_dataset = train_dataset.map(operations=train_transforms_list, input_columns=["data"])  #数据处理

    ## 5. 使用batch
    train_dataset = train_dataset.batch(8, drop_remainder=True)

    return train_dataset


if __name__ == '__main__':
    ms_train_dataset = ms_load_dataset()
    ms_dataset_batch = next(ms_train_dataset.create_dict_iterator())
    print(ms_dataset_batch.keys())
    print(ms_dataset_batch['data'].shape)
    print(ms_dataset_batch['label'])

    # ms_train_dataset.batch(64, drop_remainder=True)

    # for ms_dataset_batch in ms_train_dataset.create_dict_iterator():
    #     print(ms_dataset_batch.keys())
    #     print(ms_dataset_batch['data'].shape)
    #     print(ms_dataset_batch['label'])

    net = resnet18(pretrained=False)
    ms_optimizer = nn.SGD(net.trainable_params(), learning_rate=0.01)
    ms_net_loss = nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    ms_model = ms.train.Model(net, ms_net_loss, ms_optimizer, metrics={"Accuracy": nn.metrics.Accuracy()})
    ms_model.train(epoch=300, train_dataset=ms_train_dataset, callbacks=[loss_cb] ,dataset_sink_mode=False)

    pre = ms_model.predict()
