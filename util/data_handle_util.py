import csv
import re

'''
写入数据
'''


def write_data(filepath, data, username):
    with open(filepath, 'w', newline='') as f:
        write = csv.writer(f)
        x = re.split('!|\n', data)  # data从客户端传入
        x.append(username)  # 第451列是用户名作为label
        # x.append(date)          # 第452列是年月日标注
        write.writerow(x)


'''
以追加的方式写入数据
'''


def additional_write_data(filepath, data, username):
    with open(filepath, 'a', newline='') as f:
        write = csv.writer(f)
        x = re.split('!|\n', data)  # data从客户端传入
        x.append(username)  # 第451列是用户名作为label
        # x.append(date)          # 第452列是年月日标注
        write.writerow(x)
