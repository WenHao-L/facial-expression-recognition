import os
import json
import random
import torch
import numpy as np
from dataset import FER2013, FER2013_test, FER2013_test_without_tta
from models import resnet
from fer2013trainer import FER2013Trainer

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)


seed = 1234
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



def main(config_path):
    """
    This is the main function to make the training up

    Parameters:
    -----------
    config_path : srt
        path to config file
    """
    # load configs and set random seed
    configs = json.load(open(config_path))
    configs["cwd"] = os.getcwd()


    #2022-05-03
    model = resnet.resnet18

    #pre_model = torch.load(os.path.join("checkpoint", "resnet34_test_2022Apr04_21.05.pth"))
    #model.load_state_dict(pre_model["net"])
    #model.load_state_dict(torch.load(os.path.join("checkpoint", "resnet34_test_2022Apr04_21.05.pth")))
    # load dataset


    train_set = FER2013("Training", configs)
    val_set = FER2013("PublicTest", configs)
    # test_set = FER2013_test("PrivateTest", configs)
    test_set = FER2013_test_without_tta("PrivateTest", configs)
    #print(train_set)PublicTest PrivateTest



    # from trainers.centerloss_trainer import FER2013Trainer
    trainer = FER2013Trainer(model, train_set, val_set, test_set, configs)

    # trainer.train()

    trainer.test(checkpoint_path="checkpoint/resnet18_test_2022May24_20.53")

if __name__ == "__main__":
    main("./configs/fer2013_config.json")
