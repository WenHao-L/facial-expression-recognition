import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from data.dataset import CustomDataset


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


def get_dataloaders(path, bs=64, augment=True, fine_tuning=False):
    """ Prepare train, val, & test dataloaders
        Augment training data using:
            - cropping
            - shifting (vertical/horizental)
            - horizental flipping
            - rotation

        input: path to fer2013 csv file
        output: (Dataloader, Dataloader, Dataloader) """

    fer2013, emotion_mapping = load_data(path)

    if fine_tuning:
        xtrain, ytrain = prepare_data(fer2013[fer2013['Usage'] == 'Training' | fer2013['Usage'] == 'PrivateTest'])
    else:    
        xtrain, ytrain = prepare_data(fer2013[fer2013['Usage'] == 'Training'])
    xval, yval = prepare_data(fer2013[fer2013['Usage'] == 'PrivateTest'])
    xtest, ytest = prepare_data(fer2013[fer2013['Usage'] == 'PublicTest'])

    mu, st = 0.50786453, 0.24790359

    test_transform = transforms.Compose([
        # transforms.Scale(52),
        transforms.TenCrop(40),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda tensors: torch.stack([transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors])),
    ])

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomApply([transforms.RandomResizedCrop(48, scale=(0.8, 1.2))], p=0.5),  # output size is 48*48, rescaling the images up to ±20%
            transforms.RandomApply([transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5), # horizontally and vertically shifting the image by up to ± 20 % of its size,
            transforms.RandomApply([transforms.RandomRotation(10)], p=0.5), # rotating it up to ± 10 degrees

            transforms.TenCrop(40), # Crop 5 images of size size at the top, bottom, left, right and center of the image. Tencrop mirrors these 5 images horizontally or vertically to get 10 images. 
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),  # Since Tencrop method return a tuple, each element represents an image, we also need to convert this tuple into a tensor of an image
            transforms.Lambda(lambda tensors: torch.stack([transforms.RandomErasing(p=0.5)(t) for t in tensors])), # Randomly occlude the image
            transforms.Lambda(lambda tensors: torch.stack([transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors])), # normalize by dividing each pixel by 255
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
