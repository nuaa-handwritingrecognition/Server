# -*- coding:utf-8 -*-
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import util.constant_util

N = 17

train_path = 'G:\\handWritingRecognition\\Server\\data\\all_register.csv'

def handler(path, username):
    data1 = np.loadtxt(train_path, dtype=float, delimiter=',')  # 4 是指第 5 列
    data2 = np.loadtxt(path, dtype=float, delimiter=',').reshape(1, 451)
    # 归一化处理
    data = np.concatenate((data1, data2), axis=0)
    row_num = data.shape[0]
    for j in range(0, 450):
        mi = min(data[:, j])
        scale = max(data[:, j]) - min(data[:, j])
        data[:, j] -= mi
        if (scale != 0):
            data[:, j] /= scale

    data1 = data[0:row_num - 1, :]
    data2 = data[row_num - 1:row_num, :]
    x_train, y_train = np.split(data1, (450,), axis=1)  # axis=0 为横向切分 ， axis=1 为纵向切分
    for i in range(y_train.shape[0]):
        if str(int(y_train[i][0])) != username:
            y_train[i][0] = 1  # 不能用这个做注册
    x_test, y_test = np.split(data2, (450,), axis=1)  # axis=0 为横向切分 ， axis=1 为纵向切分

    # 卷积内核的选取
    KNN = KNeighborsClassifier(n_neighbors=N, weights='distance')
    KNN.fit(x_train, y_train)

    if (KNN.predict(x_test) == y_test):
        return util.constant_util.SUCCESS
    return util.constant_util.FAILURE

    # print("CMatrix\n", confusion_matrix(y_test, clf.predict(x_test)))
    # print("误登率：", matrix[0][1] / (matrix[0][1] + matrix[0][0]))
    # print("误杀率：", matrix[1][0] / (matrix[1][0] + matrix[1][1]))

# handler('G:\\handWritingRecognition\\Server\\data\\individual_login\\1234567-2019-5-9-22.csv', '1234567')
