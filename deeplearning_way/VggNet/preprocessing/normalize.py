# Calculate the mean and std of the dataset

import os,sys 
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.insert(0,parentdir)

import os
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from data.dataset import CustomDataset


def load_data(path):
    fer2013 = pd.read_csv(path)
    emotion_mapping = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

    return fer2013, emotion_mapping


def prepare_data(data):
    """ Prepare data for modeling
        input: data frame with labels and pixel data
        output: image and label array """

    image_array = np.zeros(shape=(len(data), 48, 48), dtype = np.uint8)
    image_label = np.array(list(map(int, data['emotion'])), dtype = np.uint8)

    for i, row in enumerate(data.index):
        image = np.fromstring(data.loc[row, 'pixels'], dtype=int, sep=' ')
        image = np.reshape(image, (48, 48))
        image_array[i] = image

    return image_array, image_label


def get_dataloaders(path, bs=64, augment=True, tag=False):
    """ Prepare train, val, & test dataloaders
        Augment training data using:
            - cropping
            - shifting (vertical/horizental)
            - horizental flipping
            - rotation

        input: path to fer2013 csv file
        output: (Dataloader, Dataloader, Dataloader) """

    fer2013, emotion_mapping = load_data(path=path)

    if tag:
        xtrain, ytrain = prepare_data(fer2013[(fer2013['Usage'] == 'Training') | (fer2013['Usage'] == 'PrivateTest')])
    else:
        xtrain, ytrain = prepare_data(fer2013[fer2013['Usage'] == 'Training'])
    xval, yval = prepare_data(fer2013[fer2013['Usage'] == 'PrivateTest'])
    xtest, ytest = prepare_data(fer2013[fer2013['Usage'] == 'PublicTest'])

    test_transform = transforms.Compose([
        transforms.Resize(40),
        transforms.ToTensor()
    ])

    if augment:
        train_transform = transforms.Compose([
        transforms.Resize(40),
        transforms.ToTensor()
        ])
    else:
        train_transform = test_transform

    train = CustomDataset(xtrain, ytrain, train_transform)
    val = CustomDataset(xval, yval, test_transform)
    test = CustomDataset(xtest, ytest, test_transform)

    trainloader = DataLoader(train, batch_size=bs, shuffle=True, num_workers=0)
    valloader = DataLoader(val, batch_size=64, shuffle=True, num_workers=0)
    testloader = DataLoader(test, batch_size=64, shuffle=True, num_workers=0)

    return trainloader, valloader, testloader

path = os.path.join(os.getcwd(), 'datasets', 'fer2013', 'fer2013_new.csv')

# Calculate the mean and std of the training set + validation set
train_loader, val_loader, test_loader = get_dataloaders(path=path, bs=32138, tag=True)
train_val_data = iter(train_loader).next()[0]
print(train_val_data.shape)
train_val_data_mean = np.mean(train_val_data.numpy(), axis=(0, 1, 2, 3))
train_val_data_std = np.std(train_val_data.numpy(), axis=(0, 1, 2, 3))
print(train_val_data_mean, train_val_data_std)


# train_loader, val_loader, test_loader = get_dataloaders(path=path, bs=28559)

# train = iter(train_loader).next()[0]
# print(train.shape)
# train_mean = np.mean(train.numpy(), axis=(0, 1, 2, 3))
# train_std = np.std(train.numpy(), axis=(0, 1, 2, 3))

# val = iter(val_loader).next()[0]
# val_mean = np.mean(val.numpy(), axis=(0, 2, 3))
# val_std = np.std(val.numpy(), axis=(0, 2, 3))

# test = iter(test_loader).next()[0]
# test_mean = np.mean(test.numpy(), axis=(0, 2, 3))
# test_std = np.std(test.numpy(), axis=(0, 2, 3))

# print(train_mean)
# print(train_std)
# print(val_mean)
# print(val_std)
# print(test_mean)
# print(test_std)
