'''
对数据归一化处理
'''
import numpy as np
import pandas as pd
import os

np.set_printoptions(suppress=True)

# 数据归一化
s_sign_all = str(os.path.dirname(os.getcwd())) + '\\data\\all_register.csv'          # 生成记录所有人注册数据的文件名
df = pd.read_csv(s_sign_all, header=None)
mat = np.array(df)
# 行数
row_num = mat.shape[0]

all_min = 99999999
all_max = -9999999
for j in range(450):
    # 压力值无需改变
    if j % 3 == 2:
        continue

    # 拿到全局最大最小值
    # todo 这个策略后期可能会出现问题
    mi = min(mat[:, j])
    all_min = min(mi, all_min)
    ma = max(mat[:, j])
    all_max = max(ma, all_min)

scale = all_max - all_min
# print(all_max)
# print(all_min)
# print(scale)

for j in range(450):
    # 压力值无需改变
    if j % 3 == 2:
        continue
    # 给所有的 x y 都加上最小值,然后除去 scale
    mat[:, j] += abs(all_min)
    if (scale != 0):
        mat[:, j] /= scale
np.savetxt('normalized.csv', mat, delimiter=',')
