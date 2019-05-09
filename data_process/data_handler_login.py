'''
对从客户端接收到的数据统一处理，登录数据生成1.csv个人数据表格，在上一级生成all.csv所有人的登录数据表格
'''

import csv
import datetime
import re
import os

def handler(username, data):
    cur = datetime.datetime.now()            #获取当前系统时间
    year = str(cur.year)
    month = str(cur.month)
    day = str(cur.day)
    hour = str(cur.hour)
    minute = str(cur.minute)
    date = year + '.' + month + '.' + day
    s = "-"
    seq = (username, year, month, day, hour)      # username从客户端传入
    s = s.join(seq)                    # 生成用-分隔的字符串序列，作为文件名
    # s_log = str(os.path.dirname(os.getcwd())) + '\\data\\individual_login\\' + s + '.csv'      # 生成记录个人登录数据的文件名
    s_log = 'G:\\handWritingRecognition\\Server\\data\\individual_login\\'+ s + '.csv'
    # s_log_all = str(os.path.dirname(os.getcwd())) + '\\data\\all_login.csv'            # 生成记录所有人登录数据的文件名
    s_log_all = 'G:\\handWritingRecognition\\Server\\data\\all_login.csv'
    # 写个人登录数据文件，只有一行，450+1+1列
    with open(s_log, 'w', newline='') as f:
        write = csv.writer(f)
        x = re.split('!|\n', data)           # data从客户端传入
        x.append(username)      # 第451列是用户名作为label
        # x.append(date)          # 第452列是年月日标注
        write.writerow(x)

    # 将个人登录数据写入all_login.csv
    with open(s_log_all, "a", newline='') as f:  # 模式是a，追加写入
        write = csv.writer(f)
        x = re.split('!|\n', data)  # data从客户端传入
        x.append(username)      # 第451列是用户名作为label
        # x.append(date)          # 第452列是年月日标注
        write.writerow(x)

    return s_log
    # return True