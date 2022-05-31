import cv2
import torch
import torchvision.transforms as transforms
import numpy as np

from PIL import Image

def face_detect(img_path):
    """
    Detect faces in test images
    :param img_path: full path to the image
    :return:
    """
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('./params/haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(
        img_gray,  # image to be detected
        scaleFactor=1.1,  # The scale factor of the search window in two consecutive scans, the default is 1.1, that is, the search window is enlarged by 10% each time
        minNeighbors=1,  # The minimum number of adjacent rectangles that constitute the detection target If the number of small rectangles that constitute the detection target and less than minneighbors - 1 will be excluded, if the minneighbors is 0, the function will return all detected candidate rectangles without doing anything
        minSize=(30, 30)
    )
    return img, img_gray, faces


def generate_faces(face_img, model):
    """
    Augment the detected face
    :param face_img: Grayscale single face image
    :param img_size: target image size
    :return:
    """
    if model == 'VggNet':
        face_img = cv2.resize(face_img, (48, 48), interpolation=cv2.INTER_LINEAR)

        face_img = Image.fromarray(face_img)

        data_transform = transforms.Compose([transforms.TenCrop(40),
                                        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        # transforms.Lambda(lambda tensors: torch.stack([transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors])),
                                    ])

    face_img = data_transform(face_img)
    return face_img


def predict_expression(img_path, model):
    """
    Expression prediction for n faces in the picture
    :param img_path:
    :return:
    """

    img, img_gray, faces = face_detect(img_path)
    face_list = list()
    # iterate over each face
    for (x, y, w, h) in faces:
        face_img_gray = img_gray[y:y + h, x:x + w]
        face_img = generate_faces(face_img_gray, model)
        face_list.append(face_img)

    return img, faces, face_list
