import os
import warnings

import torch
from torch import nn
from data.fer2013 import get_dataloaders
from utils.setup_network import setup_network
from utils.loops import evaluate_test

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


if __name__ == "__main__":
    # Important parameters
    hps = {
        'network': 'vgg',  # which network do you want to train
        'name': 'my_vgg',  # whatever you want to name your run
        'n_epochs': 150,
        'model_save_dir': os.path.join(os.getcwd(), 'checkpoints', 'my_vgg'),  # where will checkpoints be stored
        'restore_epoch': None,  # continue training from a specific saved point
        'start_epoch': 0,
        'lr': 0.01,  # starting learning rate
        'save_freq': 20,  # how often to create checkpoints
        'drop': 0.1,
        'bs': 64,
    }
    data_path = os.path.join(os.getcwd(), 'datasets', 'fer2013', 'fer2013_new.csv')
    trainloader, valloader, testloader = get_dataloaders(path=data_path, bs=hps['bs'])

    # path = os.path.join(os.getcwd(), 'checkpoints', 'my_vgg')
    # files = os.listdir(path)
    # files.remove('acc.jpg')
    # files.remove('loss.jpg')
    # print(files)

    files = ['epoch_50']  # the final model

    for file in files:
        restore_epoch = file.split('_')[1]
        # print(restore_epoch)
        hps['restore_epoch'] = int(restore_epoch)

        # build network
        logger, net = setup_network(hps)
        net = net.to(device)

        criterion = nn.CrossEntropyLoss()

        acc_test, loss_test = evaluate_test(net, testloader, criterion)
        print('Test Accuracy: %2.4f %%' % acc_test,
            'Test Loss: %2.6f' % loss_test,
            sep='\t\t')