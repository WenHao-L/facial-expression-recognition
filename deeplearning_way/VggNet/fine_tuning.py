import sys
import os
import warnings

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler

from data.fer2013 import get_dataloaders
from utils.checkpoint import save
from utils.hparams import setup_hparams
from utils.loops import train, evaluate
from utils.setup_network import setup_network

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run(net, logger, hps):
    # Create dataloaders
    path = os.path.join('datasets', 'fer2013', 'fer2013_new.csv')
    trainloader, valloader, testloader = get_dataloaders(path=path, bs=hps['bs'], fine_tuning=True)

    net = net.to(device)

    learning_rate = 0.0001
    scaler = GradScaler()

    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, nesterov=True, weight_decay=0.0001)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6, last_epoch=-1, verbose=False)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    num_epoch = 50  # num_epoch of fine tuning

    print("Training", hps['name'], "on", device)
    for epoch in range(num_epoch):

        acc_tr, loss_tr = train(net, trainloader, criterion, optimizer, scaler)
        logger.loss_train.append(loss_tr)
        logger.acc_train.append(acc_tr)

        acc_v, loss_v = evaluate(net, testloader, criterion)  # Changed to the test set for the convenience of viewing the results
        logger.loss_val.append(loss_v)
        logger.acc_val.append(acc_v)

        # Update learning rate
        scheduler.step()

        if acc_v > best_acc:
            best_acc = acc_v
            hps['lr'] = optimizer.state_dict()['param_groups'][0]['lr']  # save the learning rate

            save(net, logger, hps, epoch + 1)
            logger.save_plt(hps)

        if (epoch + 1) % hps['save_freq'] == 0:
            hps['lr'] = optimizer.state_dict()['param_groups'][0]['lr']

            save(net, logger, hps, epoch + 1)
            logger.save_plt(hps)

        if (epoch + 1) == 50:
            hps['lr'] = optimizer.state_dict()['param_groups'][0]['lr']

            save(net, logger, hps, epoch + 1)
            logger.save_plt(hps)

        print('Epoch %2d' % (epoch + 1),
              'Train Accuracy: %2.4f %%' % acc_tr,
              'Test Accuracy: %2.4f %%' % acc_v,
              sep='\t\t')


if __name__ == "__main__":
    # Important parameters
    hps = setup_hparams(sys.argv[1:])
    logger, net = setup_network(hps)

    run(net, logger, hps)