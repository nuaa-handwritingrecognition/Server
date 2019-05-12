'''
对从客户端接收到的数据写入登录数据表格
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
    s_log = 'G:\\handWritingRecognition\\Server\\data\\individual_login\\' + filename + '.csv'
    s_log_all = 'G:\\handWritingRecognition\\Server\\data\\all_login.csv'
    # 写个人登录数据文件，总文件里面用追加方式写
    util.data_handle_util.write_data(s_log, data, username)
    util.data_handle_util.additional_write_data(s_log_all, data, username)

    return s_log
