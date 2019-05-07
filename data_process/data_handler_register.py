'''
对从客户端接收到的数据统一处理
登录数据生成1.csv个人数据表格
在上一级生成all.csv所有人的登录数据表格
'''

import csv
import datetime
import re
import os


def handler(username, data):
    cur = datetime.datetime.now()  # 获取当前系统时间
    year = str(cur.year)
    month = str(cur.month)
    day = str(cur.day)
    hour = str(cur.hour)
    minute = str(cur.minute)
    date = year + '.' + month + '.' + day
    s = "-"
    seq = (username, year, month, day, hour)  # username从客户端传入
    s = s.join(seq)  # 生成用-分隔的字符串序列，作为文件名
    # s_sign = str(os.path.dirname(os.getcwd())) + '\\data\\individual_register\\' + s + '.csv'      # 生成记录个人注册数据的文件名
    s_sign = 'G:\\handWritingRecognition\\Server\\data\\individual_register\\' + s + '.csv'
    # s_sign_all = str(os.path.dirname(os.getcwd())) + '\\data\\all_register.csv'          # 生成记录所有人注册数据的文件名
    s_sign_all = 'G:\\handWritingRecognition\\Server\\data\\all_register.csv'
    # 要写10次所以用追加模式
    with open(s_sign, 'a', newline='')  as f:
        write = csv.writer(f)
        x = re.split('!|\n', data)
        x.append(username)  # 第451列是用户名作为label
        # x.append(date)  # 第452列是年月日标注
        write.writerow(x)
    # 将个人注册数据写入all_sin.csv
    with open(s_sign_all, "a", newline='') as f:  # 模式是a，追加写入
        write = csv.writer(f)
        x = re.split('!|\n', data)
        x.append(username)  # 第451列是用户名作为label
        # x.append(date)  # 第452列是年月日标注
        write.writerow(x)

    return True
