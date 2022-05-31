# Expression recognition using LBP+SVM

import os
import joblib
from tqdm import tqdm
import pandas as pd
import numpy as np
from skimage import feature as skif
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler


class LBP(object):
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

    def get_lbp(self, image):
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

    def evaluate_lbp(self, path):
        """
        Evaluation function
        :return:
        """
        # download data
        x_train, y_train, x_val, y_val, x_test, y_test, emotion_mapping = self.get_dataloaders(path)
        print(x_train.shape, x_val.shape, x_test.shape)

        # extract the LBP features of the training set
        train = []
        for i in tqdm(np.arange(0, x_train.shape[0], 1)):
            res = self.get_lbp(x_train[i])
            res = np.array(res).reshape(-1)
            res = np.append(res, y_train[i])
            train.append(res)
        train = np.array(train)
        print('train_shape:', train.shape)
        print('train[0]:', train[0])

        # extract the LBP features of the validation set
        val = []
        for i in tqdm(np.arange(0, x_val.shape[0], 1)):
            res = LBP().get_lbp(x_val[i])
            res = np.array(res).reshape(-1)
            res = np.append(res, y_val[i])
            val.append(res)
        val = np.array(val)
        print('val_shape:', val.shape)
        print('val[0]:', val[0])

        # extract the LBP features of the test set
        test = []
        for i in tqdm(np.arange(0, x_test.shape[0], 1)):
            res = LBP().get_lbp(x_test[i])
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
    path = os.path.join('tradition_way', 'datasets', 'fer2013_new.csv')
    lbp = LBP()
    lbp.evaluate_lbp(path)
    # joblib.dump(lbp.svc, 'lbp_svm.pkl')  # save the svm model
