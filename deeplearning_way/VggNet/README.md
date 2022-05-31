# VGGNet 

This work refers to [Facial Emotion Recognition: State of the Art Performance on FER2013.](https://arxiv.org/abs/2105.03588) And the code is referenced from [GitHub.](https://github.com/usef-kh/fer)

Our final model checkpoint can be found [here](https://pan.baidu.com/s/1HlzXz15wfg2VZiwqvb50xg?pwd=hi06)

<br>

## Overview
In this work, we adopt the VGGNet architecture and fine-tune its hyperparameters to achieve 72.73% single-network classification accuracy on FER2013.

<br>

## Installation
To use this repo, create a conda environment using `environment.yml` or `requirements.txt`

```
# from environment.yml (recommended)
conda env create -f environment.yml

# from requirements.txt
conda create --name <env> --file requirements.txt
```
Download the offical [fer2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) dataset, and put it in `datasets/fer2013/fer2013.csv`

<br>

## Usage

Before training your own model, preprocess the dataset to remove non-face data from the dataset, run the following
```
python preprocessing/dele_noface.py
```

To train your own version of our network, run the following

```
python train.py network=vgg name=my_vgg
```
To change the default parameters, you may also add arguments such as `bs=128` or `lr=0.1`. For more details, please refer to `utils/hparams.py`

<br>

To fine tuning your model, run the following
```
python fine_tuning.py network=vgg name=my_vgg restore_epoch=120(Choose the best model you have trained)
```

<br>

To test the accuracy of your model, run the following
```
python test.py
```
Before test, Change where your model is located, please refer to `test.py` 
