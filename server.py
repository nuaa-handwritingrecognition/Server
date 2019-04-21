# -*-coding:utf-8-*-
from flask import Flask
from flask import request
import os

app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True


# 此方法处理用户注册
@app.route('/register', methods=['POST'])
def register():
    # 客户端传来的数据
    username = request.form['username']
    password = request.form['password']
    length = request.form['length']
    data = request.form['data']
    print('username:' + username + '\n')
    print('password:' + password + '\n')
    print('length:' + length + '\n')
    print('data:' + data + '\n')
    # 返回给客户端的
    return '0'

@app.route('/')
def test():
    return '服务器正常运行'

if __name__ == '__main__':
    # 将本机设为服务器，且允许外网访问
    app.run(host='0.0.0.0')
