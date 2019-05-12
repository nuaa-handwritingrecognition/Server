# -*-coding:utf-8-*-
from flask import Flask
from flask import request
import os

import data_process.data_handler_login
import data_process.data_handler_register
import alogirithm_model.svm.realize
import alogirithm_model.lstm.realize

app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True


# 此方法处理用户注册
@app.route('/register', methods=['POST'])
def register():
    # 客户端传来的数据
    # 需要保证用户名唯一
    phone = request.form['phone']
    data = request.form['data']
    # print('data:' + data + '\n')
    # 返回给客户端的 '1' 代表成功
    user_data_path = data_process.data_handler_register.handler(phone, data)
    # 送进网络训练
    print(phone + ' register success')
    return alogirithm_model.lstm.realize.train_handler("G:\\handWritingRecognition\\Server\\data\\all_register.csv",
                                                       phone)


# 此方法处理用户注册
@app.route('/login', methods=['POST'])
def login():
    # 客户端传来的数据
    phone = request.form['phone']
    data = request.form['data']
    # print('data:' + data + '\n')
    # 返回给客户端的 '1' 代表成功
    # 处理数据
    user_data_path = data_process.data_handler_login.handler(phone, data)
    # svm 方法预测
    # return alogirithm_model.svm.realize.handler(user_data_path, username)
    # lstm 方法预测
    # return alogirithm_model.lstm.realize.test_handler(user_data_path,username)
    print(phone + ' login success')
    return '1'


@app.route('/')
def test():
    return '服务器正常运行'


if __name__ == '__main__':
    # 将本机设为服务器，且允许外网访问
    app.run(host='0.0.0.0')
