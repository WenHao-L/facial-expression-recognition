# Application of traditional machine learning methods in facial expression recognition


## Overview
We use the **LBP** and **Gabor** algorithm to extract face features, and use the collected features to train the **svm** classifier, where **LBP+SVM** achieves **41.85%** test set accuracy on **fer2013 dataset**, and **Gabor+SVM** achieves **42.75%** accuracy.

But this accuracy rate is not ideal. We guess that because the LBP and Gabor algorithm cannot extract as many features of the face as possible.

So we **replace the LBP feature extraction with a neural network and refer to [VGGNet](https://paperswithcode.com/paper/facial-emotion-recognition-state-of-the-art)**. That is, the trained VggNet is used to extract features, and the extracted features are used to train the svm classifier. 

The change is only to **remove the output layer of VggNet**, and the output of the neural network becomes the features.

Fortunately, the modified model achieved an accuracy rate of **over 70%** on the test set (which is almost equivalent to using VggNet alone), which further verifies that our conjecture is correct.

<br>

## Installation
Download the offical [fer2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) dataset, and place it in the outmost folder with the following folder structure.

**"./datasets/fer2013.csv"**

<br>

The VggNet model checkpoint can be found [here](https://pan.baidu.com/s/1HlzXz15wfg2VZiwqvb50xg?pwd=hi06) (of course you can also train your own model), and place it in the outmost folder with the following folder structure.

**"./params/"**

<br>

## Usage

Before training your own model, preprocess the dataset to remove non-face data from the dataset, run the following
```
python preprocessing/dele_noface.py
```
To train your own version of LBP+SVM, run the following
```
python LBP_SVM.py
```

To train your own version of Gabor+SVM, run the following
```
python Gabor_SVM.py
```

To train your own version of VGG+SVM, run the following
```
python VGG_SVM.py
```