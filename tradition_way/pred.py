import cv2
import joblib
from PIL import Image
import numpy as np
from skimage import feature as skif
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


def get_lbp(image):
    """
    Get the LBP of a given image and divide it into several regions
    :param rgb:
    :return:
    """
    gridx = 6
    gridy = 6
    widx = 8
    widy = 8
    hists = []
    for i in range(gridx):
        for j in range(gridy):
            mat = image[i * widx: (i + 1) * widx, j * widy: (j + 1) * widy]
            lbp = skif.local_binary_pattern(mat, 8, 1, 'uniform')
            max_bins = 10
            hist, _ = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))
            hists.append(hist)
    out = np.array(hists).reshape(-1, 1)
    return out


def image_preprocess(face_img):
    '''
    face_img: 接收传入的人脸图片
    '''
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.resize(face_img, (48, 48), interpolation=cv2.INTER_LINEAR)
    face_array = np.array(face_img)
    # print(face_array.shape)

    return face_array


def feature_extraction(face_array):
    '''
    特征提取, 先lbp特征提取, 再pca特征降维
    '''
    lbp_feature = get_lbp(face_array)
    feature = lbp_feature.reshape(1, -1)
    # print(feature.shape)

    scaler = MinMaxScaler()
    scaler.fit(feature)
    feature = scaler.transform(feature)

    pca = joblib.load('pca.pkl')
    feature = pca.transform(feature)

    return feature


def main():
    img_path = 'surprised1.jpg'
    image = cv2.imread(img_path)
    image_array = image_preprocess(image)
    face_feature = feature_extraction(image_array)

    model = joblib.load('lbp_svm.pkl')
    pred = model.predict(face_feature)

    print(pred)


if __name__ == '__main__':
    main()