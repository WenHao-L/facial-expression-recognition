# Expression recognition using VggNet+SVM

import os
import warnings
import torch
import torchvision.transforms as transforms
import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm
from VGG import Vgg
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_data(path):
    fer2013 = pd.read_csv(path)
    emotion_mapping = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

    return fer2013, emotion_mapping


def prepare_data(data):
    """ Prepare data for modeling
        input: data frame with labels and pixel data
        output: image and label array """

    image_array = np.zeros(shape=(len(data), 48, 48), dtype=np.uint8)
    image_label = np.array(list(map(int, data['emotion'])), dtype=np.uint8)

    for i, row in enumerate(data.index):
        image = np.fromstring(data.loc[row, 'pixels'], dtype=int, sep=' ')
        image = np.reshape(image, (48, 48))
        image_array[i] = image

    return image_array, image_label


def get_dataloaders(path):
    """ Prepare train, val, & test dataloaders
        input: path to fer2013 csv file
        output: (Dataloader, Dataloader, Dataloader) """

    fer2013, emotion_mapping = load_data(path)

    # If use all the data of the training set for feature extraction, you need to pay attention to the computer memory may not be enough.
    # of course, you don't need to worry if your computer performance is good enough
    x_train, y_train = prepare_data(fer2013[fer2013['Usage'] == 'Training'])
    x_val, y_val = prepare_data(fer2013[fer2013['Usage'] == 'PrivateTest'])
    x_test, y_test = prepare_data(fer2013[fer2013['Usage'] == 'PublicTest'])

    return x_train, y_train, x_val, y_val, x_test, y_test, emotion_mapping


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
    svc = SVC(kernel='rbf', gamma='scale')
    scaler = MinMaxScaler()
    x_train, y_train = train_data[:, :train_data.shape[1] - 1], train_data[:, -1]
    x_test, y_test = test_data[:, :test_data.shape[1] - 1], test_data[:, -1]
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    new_train, new_test = pca(x_train, x_test, min(1000, min(x_train.shape[1] - 7, x_train.shape[0])))
    print('new_train.shape = ', new_train.shape)
    print('new_test.shape = ', new_test.shape)

    svc.fit(new_train, y_train)
    pred = svc.predict(new_test)

    # 获得识别率
    recognition_rate = np.sum((pred == y_test)) / len(test_data[:, -1])
    print(recognition_rate)


if __name__ == '__main__':
    vgg_model_path = os.path.join('params', 'VGGNet')  # Path to the trained model file
    data_path = os.path.join('datasets', 'fer2013.csv')

    model = Vgg()
    checkpoint = torch.load(vgg_model_path)
    model.load_state_dict(checkpoint['params'], strict=False)
    model = model.to(device)

    x_train, y_train, x_val, y_val, x_test, y_test, emotion_mapping = get_dataloaders(data_path)

    mu, st = 0.50786453, 0.24790359
    # train = []
    for i in tqdm(np.arange(0, x_train.shape[0], 1)):
        face_img = Image.fromarray(x_train[i])

        data_transform = transforms.Compose([transforms.TenCrop(40),
                                            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                            transforms.Lambda(lambda tensors: torch.stack([transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors])),
                                             ])

        face_img = data_transform(face_img)
        # face_img = torch.unsqueeze(face_img, 0)
        # extract features with VggNet
        face_img = face_img.to(device)
        feature = model(face_img)
        feature = np.array(feature.cpu().detach().numpy()).reshape(-1)
        feature = np.append(feature, y_train[i])
        feature = feature[np.newaxis, :]
        if i == 0:
            train = feature
        else:
            train = np.append(train, feature, axis=0)
        # print(train.shape)
        # train.append(feature)
    # train = np.array(train)

    # val = []
    for i in tqdm(np.arange(0, x_val.shape[0], 1)):
        face_img = Image.fromarray(x_val[i])
        data_transform = transforms.Compose([transforms.TenCrop(40),
                                            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                            transforms.Lambda(lambda tensors: torch.stack([transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors])),
                                             ])

        face_img = data_transform(face_img)
        # face_img = torch.unsqueeze(face_img, 0)
        face_img = face_img.to(device)
        feature = model(face_img)
        feature = np.array(feature.cpu().detach().numpy()).reshape(-1)
        feature = np.append(feature, y_val[i])
        feature = feature[np.newaxis, :]
        if i == 0:
            val = feature
        else:
            val = np.append(val, feature, axis=0)
        # val.append(feature)
    # val = np.array(val)

    # test = []
    for i in tqdm(np.arange(0, x_test.shape[0], 1)):
        face_img = Image.fromarray(x_test[i])
        data_transform = transforms.Compose([transforms.TenCrop(40),
                                            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                            transforms.Lambda(lambda tensors: torch.stack([transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors])),
                                             ])

        face_img = data_transform(face_img)
        # face_img = torch.unsqueeze(face_img, 0)
        face_img = face_img.to(device)
        feature = model(face_img)
        feature = np.array(feature.cpu().detach().numpy()).reshape(-1)
        feature = np.append(feature, y_test[i])
        feature = feature[np.newaxis, :]
        if i == 0:
            test = feature
        else:
            test = np.append(test, feature, axis=0)
        # test.append(feature)
    # test = np.array(test)

    # train the svm classifier and evaluate the model
    print("Eval_acc:")
    SVM(train, val)

    print("Test_acc:")
    SVM(train, test)
