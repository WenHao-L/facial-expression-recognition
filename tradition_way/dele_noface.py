# Remove non-face data from training and validation sets

import os
import pandas as pd
import numpy as np
# import matplotlib.plt as plt

cwd = os.getcwd()
path = os.path.join(cwd, 'datasets', 'fer2013.csv')
data = pd.read_csv(path)

pixels = data["pixels"].tolist()
# print('data_lenth', len(pixels))

# read the tag
label_path = os.path.join(cwd, 'datasets', 'fer_plus.csv')
label = pd.read_csv(label_path)
# print('data_len', len(label))

no_face = label["NF"]
# print(no_face)

noface = np.asarray(no_face)
noface_where = np.where(noface==10)
# print(noface_where)
noface_where = np.asarray(noface_where)
noface_where = np.squeeze(noface_where)
print(noface_where)
print(len(noface_where))

# Extract the sequence numbers of images marked with errors in the training set and test set
train_val_noface_where = np.append(noface_where[0:150], noface_where[160::])
print(len(train_val_noface_where))

data_new = data.drop(train_val_noface_where, axis=0)
print(len(data_new))
new_path = os.path.join(cwd, 'datasets', 'fer2013_new.csv')
data_new.to_csv(new_path, index=False)

## show some non-face data
# plt.figure()
# img_list = [59, 2059, 2171, 2809, 5722, 6458]
# for i in range(len(img_list)):
#     img = pixels[img_list[i]]    #str
#     img = list(map(int, img.split(" ")))
#     img = np.asarray(img).reshape(48, 48)   #48*48

#     plt.subplot(2,3,i+1)
#     plt.imshow(img)
# plt.show()