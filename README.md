# Facial Expression Recognition (FER)
This work is about facial expression recognition.

## Overview
In this work, we explore the application of traditional machine learning methods and deep learning methods in the field of facial expression recognition.

In the field of traditional machine learning, we use **LBP** and **Gabor** to extract facial expression features, and train an **SVM** classifier through the collected features.

**LBP+SVM** achieves **41.85%** test set classification accuracy on the **fer2013 dataset**, while **Gabor+SVM** achieves **42.75%** classification accuracy.

In the field of deep learning, we use **[VGGNet](https://paperswithcode.com/paper/facial-emotion-recognition-state-of-the-art)** and achieve **72.73%** test set accuracy on the fer2013 dataset.

We can find that traditional machine learning methods have achieved limited success in facial expression recognition.

We guess that this is because **the LBP and Gabor algorithms cannot extract the facial expression features very well**. So we combine deep learning and traditional machine learning algorithms, use the above trained VggNet model for feature extraction, and train a new SVM classifier.

Fortunately, the modified **VGGNet+SVM** model achieved over 70% accuracy on the test set, proving our conjecture correct.

We also study the performance of **ResNet** and **ResNet+SVM** on facial expression recognition.

In addition, we wrote a simple **GUI** program for facial expression recognition, which deployed the trained VggNet model.

<br>

## Folder introduction
```
|-- tradition_way 
    (Include LBP+SVM, Garbor+SVM and VGGNet+SVM)

|-- deeplearning_way
    (Include VGGNet and ResNet)

|-- GUI
    (A simple facial expression recognition GUI)
```

<br>

## Usage
We have written a README file for each subfolder, you can go to the README file of the corresponding subfolder to see how to use our code.