'''
设置label 并将所有数据合并到一个 csv 文件中
'''
import os
import pandas as pd
import glob
import numpy as np
import tqdm


# 清理数据并保存文件
def clean(src_path, obj_path, num):
    # 返回 ?*150 的 numpy 数组
    vector = transform(src_path, num)
    # 行数
    cnt = row_num = vector.shape[0]
    for row in range(row_num):
        sum = 0
        for i in range(2, 448, 3):
            # 当sum=0时，代表第3n列（压力）全部相等。
            sum += (vector[row][i + 3] - vector[row][i])
        # 压力值全相等或者大于1，为废弃数据
        if (sum == 0) or (vector[row][2] >= 1):
            vector[row][450] = -1
            cnt -= 1
    # 新建一个矩阵vector_cleaned用来存放清洗后的数据
    vector_cleaned = np.zeros([cnt, 451])
    i = 0
    for row in range(row_num):
        if (vector[row][450] != -1):
            vector_cleaned[i] = vector[row]
            i += 1
    # 保存为指定路径
    np.savetxt(str(obj_path), vector_cleaned, delimiter=',')


# 将?*3的数据转化为?*150的数据，返回numpy数组
def transform(src_path, num):
    df = pd.read_csv(src_path, header=None)
    row_num = df.shape[0] // 150
    vector = np.zeros([row_num, 451])
    # global label
    for i in tqdm.tqdm(range(row_num)):
        for j in range(150):
            # 合并操作，并在最后一行添加标签0代表弗负类
            vector[i][3 * j: 3 * (j + 1)] = df.iloc[j + i * 150][0: 3]
            vector[i][450] = num
    # label += 1
    return vector


# 将所有用户的数据合并成一个csv文件
def merge(src_path, obj_path):
    # 查看同文件夹下的csv文件数，将csv文件列表存在csv_list中
    csv_list = glob.glob(str(src_path) + '\*.csv')
    print(u'共发现%s个CSV文件' % len(csv_list))
    # 判断文件是否存在，防止重复添加
    if os.path.exists(obj_path) == False:
        # 循环读取同文件夹下的csv文件
        for i in csv_list:
            fr = open(i, 'r').read()
            with open(str(obj_path), 'a') as f:
                f.write(fr)
        print(u'合并完毕！')


label = 1
path = './Dataset/'
# 获取该目录下所有文件，存入列表中
f = os.listdir(path)
i = 0
for j in f:
    # 拿到每一个文件名
    src_path = path + f[i]
    print('当前处理:' + src_path)
    clean(src_path, './handler_data_set/' + f[i], i + 1)
    i += 1

merge('./handler_data_set/', 'handled_data.csv')
