# -*- coding:utf-8 -*-
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix

def run(path):

    data = np.loadtxt(path, dtype=float, delimiter=',') # 4 是指第 5 列
    x, y = np.split(data, (450,), axis=1) #axis=0 为横向切分 ， axis=1 为纵向切分
    print(x)
    print(y)
    #x = x[:, :2]      #取整行，前两列
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.8)
    #clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
    #clf = svm.SVC(kernel='sigmoid')
    clf = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
    clf.fit(x_train, y_train.ravel()) #用训练数据拟合分类器模型

    print ('训练集准确率：', accuracy_score(y_train, clf.predict(x_train)))
    print ('测试集准确率：', accuracy_score(y_test, clf.predict(x_test)))

    print( "Recall", recall_score(y_test, clf.predict(x_test)))
    print("Precision", precision_score(y_test, clf.predict(x_test)))
    print("CMatrix\n", confusion_matrix(y_test, clf.predict(x_test)))

run('../../data/data_transformed.csv')