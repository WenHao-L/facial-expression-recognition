# Expression recognition using LBP+SVM

import os
import cv2
import joblib
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler


class Gabor(object):

    def gray_norm(self, img):
        """
        Grayscale normalization
        :param img:
        :return:
        """
        min_value = np.min(img)
        max_value = np.max(img)
        if max_value == min_value:
            return img
        (n, m) = img.shape
        for i in range(n):
            for j in range(m):
                img[i, j] = np.int8(255 * (img[i][j] - min_value) / (max_value - min_value))
        return img

    def normailiztaion(self, img, dets, shape_list):
        """
        Image scale grayscale normalization
        :param img:
        :param dets:
        :param shape_list:
        :return:
        """
        # Grayscale normalization
        img = self.gray_norm(img)

        # scale normalization
        img_list = []
        pt_pos_list = []
        for index, face in enumerate(dets):
            left = face.left()
            top = face.top()
            right = face.right()
            bottom = face.bottom()
            img1 = img[top:bottom, left:right]
            size = (48, 48)
            img1 = cv2.resize(img1, size, interpolation=cv2.INTER_LINEAR)

            pos = []
            for _, pt in enumerate(shape_list[index].parts()):
                pt_pos = (int((pt.x - left) / (right - left) * 90), int((pt.y - top) / (bottom - top) * 100))
                pos.append(pt_pos)
                cv2.circle(img1, pt_pos, 2, (255, 0, 0), 1)
            pt_pos_list.append(pos)
            img_list.append(img1)
        return img_list, pt_pos_list

    def adaptive_histogram_equalization(self, img):
        """
        Adaptive Histogram Equalization
        :param img:
        :return:
        """
        img.dtype = 'uint8'
        # create a CLAHE object (Arguments are optional)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        return img

    def load_data(self, path):
        fer2013 = pd.read_csv(path)
        emotion_mapping = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

        return fer2013, emotion_mapping

    def prepare_data(self, data):
        """ Prepare data for modeling
            input: data frame with labels and pixel data
            output: image and label array """

        image_array = np.zeros(shape=(len(data), 48, 48))
        image_label = np.array(list(map(int, data['emotion'])))

        for i, row in enumerate(data.index):
            image = np.fromstring(data.loc[row, 'pixels'], dtype=int, sep=' ')
            image = np.reshape(image, (48, 48))
            image_array[i] = image

        return image_array, image_label

    def get_dataloaders(self, path):
        """ Prepare train, val, & test dataloaders
            input: path to fer2013 csv file
            output: (Dataloader, Dataloader, Dataloader) """

        fer2013, emotion_mapping = self.load_data(path)

        x_train, y_train = self.prepare_data(fer2013[fer2013['Usage'] == 'Training'])
        x_val, y_val = self.prepare_data(fer2013[fer2013['Usage'] == 'PrivateTest'])
        x_test, y_test = self.prepare_data(fer2013[fer2013['Usage'] == 'PublicTest'])

        return x_train, y_train, x_val, y_val, x_test, y_test, emotion_mapping

    def build_filters(self):
        """
        Building a Gabor filter
        :return:
        """
        filters = []
        ksize = [3, 5, 7, 9]  # gabor scale, 6
        lamda = np.pi / 2.0  # wavelength
        for theta in np.arange(0, np.pi, np.pi / 4):  # gabor direction, 0°, 45°, 90°, 135°, a total of four
            for K in range(4):
                kern = cv2.getGaborKernel((ksize[K], ksize[K]), 0.56 * ksize[K], theta, lamda, 0.5, 1, ktype=cv2.CV_32F)
                kern /= 1.5 * kern.sum()
                filters.append(kern)
        return filters

    def process(self, img, filters):
        accum = np.zeros_like(img)
        for kern in filters:
            img.dtype = 'uint8'
            fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
            np.maximum(accum, fimg, accum)
        return accum

    def mean_pooling(self, img, size):
        """
        :param img:
        :param size:
        :return:
        """
        n, m = img.shape
        fn, fm = int(n / size), int(m / size)
        fimg = np.zeros((fn, fm), dtype=float)
        for i in range(fn):
            for j in range(fm):
                sum = 0
                for x in range(i * size, i * size + size):
                    for y in range(j * size, j * size + size):
                        sum += img[x, y]
                fimg[i, j] = sum / (size * size)
        return fimg

    def getGabor(self, img, filters, pic_show=False, reduction=1):
        res = []  # filter result
        for i in range(len(filters)):
            res1 = self.process(img, filters[i])
            res1 = self.mean_pooling(res1, reduction)
            # print(res1.shape)
            res.append(np.asarray(res1))
        if pic_show:
            plt.figure(2)
            for temp in range(len(filters)):
                plt.subplot(4, 4, temp + 1)
                plt.imshow(filters[temp], cmap='gray')
            plt.show()

        return res  # Return the filtering result, the result is 24 pictures, arranged according to the gabor angle

    def PCA(self, train, test, n=1000):
        from sklearn.decomposition import PCA

        pca = PCA(n_components=n)
        train = pca.fit_transform(train)
        test = pca.transform(test)
        return train, test

    def SVM(self, train_data, test_data):
        """
        :param train_data:
        :param test_data:
        :return:
        """
        self.svc = SVC(kernel='rbf', gamma='scale')
        scaler = MinMaxScaler()
        x_train, y_train = train_data[:, :train_data.shape[1] - 1], train_data[:, -1]
        x_test, y_test = test_data[:, :test_data.shape[1] - 1], test_data[:, -1]
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

        new_train, new_test = self.PCA(x_train, x_test, min(1000, min(x_train.shape[1] - 7, x_train.shape[0])))

        self.svc.fit(new_train, y_train)
        pred = self.svc.predict(new_test)

        # get recognition rate
        recognition_rate = np.sum((pred == y_test)) / len(test_data[:, -1])
        print(recognition_rate)

    def evaluate_gabor(self, path):
        """
        Evaluation function
        :return:
        """
        # download data
        x_train, y_train, x_val, y_val, x_test, y_test, emotion_mapping = self.get_dataloaders(path)

        filters = self.build_filters()

        # extract the LBP features of the training set
        train = []
        for i in tqdm(np.arange(0, x_train.shape[0], 1)):
            x_train[i] = self.gray_norm(x_train[i])
            x_tmp = self.adaptive_histogram_equalization(x_train[i])
            res = self.getGabor(x_tmp, filters, False, 8)
            res = np.array(res).reshape(-1)
            res = np.append(res, y_train[i])
            train.append(res)
        train = np.array(train)
        print('train_shape:', train.shape)
        print('train[0]:', train[0])

        # extract the LBP features of the validation set
        val = []
        for i in tqdm(np.arange(0, x_val.shape[0], 1)):
            x_val[i] = self.gray_norm(x_val[i])
            x_tmp = self.adaptive_histogram_equalization(x_val[i])
            res = self.getGabor(x_tmp, filters, False, 8)
            res = np.array(res).reshape(-1)
            res = np.append(res, y_val[i])
            val.append(res)
        val = np.array(val)
        print('val_shape:', val.shape)
        print('val[0]:', val[0])

        # extract the LBP features of the test set
        test = []
        for i in tqdm(np.arange(0, x_test.shape[0], 1)):
            x_test[i] = self.gray_norm(x_test[i])
            x_tmp = self.adaptive_histogram_equalization(x_test[i])
            res = self.getGabor(x_tmp, filters, False, 8)
            res = np.array(res).reshape(-1)
            res = np.append(res, y_test[i])
            test.append(res)
        test = np.array(test)
        print('test_shape:', test.shape)
        print('test[0]:', test[0])

        print("Eval_acc:")
        self.SVM(train, val)
        print("Test_acc:")
        self.SVM(train, test)


if __name__ == '__main__':
    path = os.path.join('datasets', 'fer2013_new.csv')
    gabor = Gabor()
    gabor.evaluate_gabor(path)
    # joblib.dump(gabor.svc, 'lbp_svm.pkl')  # save the svm model
