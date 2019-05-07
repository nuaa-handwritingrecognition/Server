import numpy as np  # 快速操作结构数组的工具
from sklearn.neighbors import KNeighborsClassifier, KDTree  # 导入knn分类器
from sklearn.model_selection import train_test_split


def Trainmatrix(filename):  # 训练集变成矩阵
    # 打开文件读取出来变成矩阵
    with open(filename, 'r') as fr:
        # 读取文件所有内容
        arrayOLines = fr.readlines()  # 每次读一行
        # 得到文件行数
        # print(len(arrayOLines))
        numberOfLines = len(arrayOLines)
        # 返回的NumPy矩阵,解析完成的数据:numberOfLines/150行(一个字是1行),300列,一个字是矩阵的一行
        returnMat = np.zeros((numberOfLines, 450))
        # 行的索引值
        # print(numberOfLines//150)
        index = 0
        flag = -1
        i = 0
        j = 3
        for line in arrayOLines:
            for j in (3, 450):
                if flag != index:
                    # s.strip(rm)，当rm空时,默认删除空白符(包括'\n','\r','\t',' ')
                    line = line.strip()
                    # 使用s.split(str="",num=string,cout(str))将字符串根据'\t'分隔符进行切片。
                    listFromLine = line.split('\t')
                    # 将数据前两列提取出来,存放到returnMat的NumPy矩阵中,也就是特征矩阵
                returnMat[index, i:j] = listFromLine[i:j]
                i = j  # 从j后面一个元素开始
                j += 3
                if flag != index:
                    flag = index
                if j == 453:
                    index += 1
                    i = 0
                    j = 3  # 赋初值
    return returnMat


def Testmatrix(filename1):  # 测试数据变成矩阵
    # 打开文件读取出来变成矩阵
    with open(filename1, 'r') as fr:
        # 读取文件所有内容
        arrayOLines = fr.readlines()  # 每次读一行
        # 得到文件行数
        # print(len(arrayOLines))
        numberOfLines = len(arrayOLines)
        # 返回的NumPy矩阵,解析完成的数据:numberOfLines/150行(一个字是1行),300列,一个字是矩阵的一行
        returnMat = np.zeros((numberOfLines // 150, 300))
        # 行的索引值
        index = 0
        i = 0
        j = 2
        for line in arrayOLines:
            # s.strip(rm)，当rm空时,默认删除空白符(包括'\n','\r','\t',' ')
            line = line.strip()
            # 使用s.split(str="",num=string,cout(str))将字符串根据'\t'分隔符进行切片。
            listFromLine = line.split('\t')
            # 将数据前两列提取出来,存放到returnMat的NumPy矩阵中,也就是特征矩阵
            returnMat[index, i:j] = listFromLine[0:2]
            i = j  # 从j后面一个元素开始
            j += 2
            if j == 302:
                index += 1
                i = 0
                j = 2  # 赋初值
    return returnMat


filename = "11.txt"
# filename1 = "测试集(前20是正类).txt"
datingDataMat = Trainmatrix(filename)
# 1为正类
y = []
for a in range(0, 121):
    y.append(0)
for a in range(121, 151):
    y.append(1)

X_train, X_test, y_train, y_test = train_test_split(datingDataMat, y, test_size=0.2)
'''
for i in range(0,101):
    datingDataMat[i,300] = 1
#0为负类
for i in range(101,227):
    datingDataMat[i,300] = 0
#print(datingDataMat)

X = datingDataMat[:,0:300]
y = datingDataMat[:,300]
'''

knn = KNeighborsClassifier(n_neighbors=2,
                           weights='distance')  # 初始化一个knn模型，设置k=2。weights='distance'样本权重等于距离的倒数。'uniform'为统一权重
knn.fit(X_train, y_train)  # 根据样本集、结果集，对knn进行建模

# datatest = Testmatrix(filename1)
# print(datatest)
result = knn.predict(X_test)  # 使用knn对新对象进行预测
# print(result)
a = [x for x in y_test if x in result]  # 两个列表表都存在
print(len(a) / len(y_test))
