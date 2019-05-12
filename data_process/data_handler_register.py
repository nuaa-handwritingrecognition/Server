'''
对从客户端接收到的数据写入注册数据表格
'''

import datetime
import util.data_handle_util


def handler(username, data):
    cur = datetime.datetime.now()  # 获取当前系统时间
    delimiter = "-"
    # 备份用户数据
    seq = (username, str(cur.year), str(cur.month), str(cur.day), str(cur.hour),str(cur.minute),str(cur.second))
    filename = delimiter.join(seq)  # 生成用-分隔的字符串序列，作为文件名
    # 用相对路径的话，服务部署上去会出错
    s_sign = 'G:\\handWritingRecognition\\Server\\data\\individual_register\\' + filename + '.csv'
    s_sign_all = 'G:\\handWritingRecognition\\Server\\data\\all_register.csv'
    # 写个人注册数据文件,因为要写10次，所以都用追加的方式写
    util.data_handle_util.additional_write_data(s_sign, data, username)
    util.data_handle_util.additional_write_data(s_sign_all, data, username)

    return s_sign
