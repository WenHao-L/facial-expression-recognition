'''
把resnet最后一层全连接层去掉，然后输出维度为512的特征保存为npy,
再把npy和对应的label，用svm进行训练
'''

import os
import json
from re import S
import torch
import numpy as np
import torch.nn.functional as F

from torch.utils.data import DataLoader

from tqdm import tqdm



from dataset import FER2013, FER2013_test
from models import resnet
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config_path = "./configs/fer2013_config.json"
checkpoint_path ="svm/checkpoint/resnet34-May25_00.23" #已经训练好的模型checkpoint
configs = json.load(open(config_path))
configs["cwd"] = os.getcwd()
# train_set = FER2013("train", configs)

train_set = FER2013("Training", configs,svm = True)
print("train_set.data_size = ",train_set.data_size)
val_set = FER2013("PublicTest", configs)
print("val_set.data_size = ",val_set.data_size)
test_set = FER2013("PrivateTest", configs,svm = True)
print("test_set.data_size = ",test_set.data_size)


train_loader = DataLoader(
            train_set,
            batch_size=32,
            num_workers=1,
            pin_memory=True, # pin_memory就是锁页内存，创建DataLoader时，设置pin_memory=True，则意味着生成的Tensor数据最开始是属于内存中的锁页内存，这样将内存的Tensor转义到GPU的显存就会更快一些。
            shuffle=False,
        )

test_loader = DataLoader(
            test_set,
            batch_size=32,
            num_workers=1,
            pin_memory=True, # pin_memory就是锁页内存，创建DataLoader时，设置pin_memory=True，则意味着生成的Tensor数据最开始是属于内存中的锁页内存，这样将内存的Tensor转义到GPU的显存就会更快一些。
            shuffle=False,
        )

val_loader = DataLoader(
            val_set,
            batch_size=32,
            num_workers=1,
            pin_memory=True, # pin_memory就是锁页内存，创建DataLoader时，设置pin_memory=True，则意味着生成的Tensor数据最开始是属于内存中的锁页内存，这样将内存的Tensor转义到GPU的显存就会更快一些。
            shuffle=False,
        )


def data_pre(data_loader,save_path=''):
    '''
    把数据集转换为npy,把原来的输入48*48转换为特征512
    '''
    input = np.empty(shape=(0,256))
    label = np.empty(shape=(0,1))
    model = resnet.resnet34() #把fc_linear全部注释掉才能加载模型
    state = torch.load(checkpoint_path)

    model.load_state_dict(state["net"], strict=False)
    #print(model)

    model = model.to(device)

    model.eval()
    with torch.no_grad():
        for i, (images, targets) in tqdm(
            enumerate(data_loader), total=len(data_loader), leave=False
        ):

            #images.to(device)
            images = images.cuda(non_blocking=True)
            bs, c, h, w = images.shape
            images = images.view(-1,c,h,w)
            # targets = torch.repeat_interleave(targets, repeats=3, dim = 0)

            # images.to(device)

            outputs = model(images)  # 10 1 200 200
            # outputs = outputs.numpy()
            outputs = outputs.cpu().detach().numpy()
            # outputs = outputs.reshape((1,512))
            input = np.vstack((input,outputs))



            targets = targets.numpy()
            # targets.resize(10,1)
            targets = targets.reshape((bs,1))
            label = np.vstack((label,targets))


            #print(type(outputs))  #tensor  shape: batch_size * 512
    npy = np.hstack((input,label))

    np.save(save_path,npy)

    return npy
    #return input,label
"""
def label2data(data_loader):
    label = np.empty(shape=(0,1))
    with torch.no_grad():
        for i, (images, targets) in tqdm(
            enumerate(data_loader), total=len(data_loader), leave=False
        ):

            # outputs = model(images)
            # outputs = outputs.numpy()
            # input = np.vstack((input,outputs))
            #print(type(outputs))  #tensor  shape: batch_size * 512
            targets = targets.numpy()
            targets = targets.reshape((1,1))
            label = np.vstack((label,targets))
    print(label.shape)
    return label
"""

from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
def pca(train, test, n=1000):

    pca = PCA(n_components=n)
    train = pca.fit_transform(train)
    test = pca.transform(test)
    return train, test

def SVM(train_data, test_data):
    """
    :param train_data:
    :param test_data:
    :return:
    """
    # svc = SVC(kernel='rbf', gamma='scale')
    svc = SVC(kernel='rbf', gamma='scale')
    scaler = MinMaxScaler()
    x_train, y_train = train_data[:, :train_data.shape[1] - 1], train_data[:, -1]
    x_test, y_test = test_data[:, :test_data.shape[1] - 1], test_data[:, -1]
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # new_train, new_test = pca(x_train, x_test, min(1000, min(x_train.shape[1] - 7, x_train.shape[0])))
    new_train, new_test = pca(x_train, x_test,'mle')

    svc.fit(new_train, y_train)
    pred = svc.predict(new_test)

    # 获得识别率
    recognition_rate = np.sum((pred == y_test)) / len(test_data[:, -1])
    print(recognition_rate)

if __name__ == '__main__':
    # train_np = main(train_loader)
    # np.save("train.npy",train_np)
    # test_np = main(test_loader)
    # np.save("test.npy",test_np)

    # train_label = label2data(train_loader)
    # #np.save("test_label.npy",test_label)
    # train_input = np.load("train.npy")
    # train = np.hstack((train_input,train_label))
    # print(train.shape)
    # np.save("train_all.npy",train)

    # val_input,val_label = main(val_loader)
    # val = np.hstack((val_input,val_label))
    # print(val.shape)
    # np.save("val_all.npy",val)

    # train = np.load('train_1.npy')
    # test = np.load('test_1.npy')

    #train = train[::10,:]
    # test = test[::10,:]

    ## 准备数据，保存为npy,方便下次直接使用
    # train = data_pre(train_loader,"train_18_conv.npy")
    # test = data_pre(test_loader,"test_18_conv.npy")
    # val = data_pre(val_loader,"val_18_conv.npy")



    # train = np.load("train_18_conv.npy")
    # val = np.load("val_18_conv.npy")


    # train_val = np.vstack((train,val))
    # np.save("train_val_18_conv.npy",train_val)

    test = np.load('test_34.npy')

    train_val = np.load("train_val_34.npy")
    # print(train_val.shape)

    # train_val = np.load("train_val_1.npy")
    print("Test_acc:")
    # val = data_pre(val_loader,'non')
    # test = data_pre(test_loader,'non')

    SVM(train_val,test)

    # print("val_acc:")
    # #train = np.load('train_all.npy')
    # val = np.load('val_all.npy')
    # SVM(train,val)