# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

# 参数
N_classes = 2  # 类别个数（二分类）
time_step = 150  # 一共有150个时间戳
input_size = 3  # 每个时间戳传入3个数据

# 超参数
learning_rate = 0.01  # 学习速率
num_units = 100  # 隐藏层神经元个数
iterations = 250  # 迭代次数
batch_size = 50  # 每次传入50个

all_register = "./../../data/all_register.csv"
test_src_path = './../../data/individual_login/7654321-2019-5-10-22.csv'


# 读取csv文件
def read_csv(src_path, username):
    data1 = np.loadtxt(src_path, dtype=float, delimiter=',').reshape(-1, 451)  # 4 是指第 5 列
    # print(data1)
    # print(data1.shape)
    x_train, y_train = np.split(data1, (450,), axis=1)  # axis=0 为横向切分 ， axis=1 为纵向切分
    for i in range(y_train.shape[0]):
        if str(int(y_train[i][0])) != username:
            y_train[i][0] = 0  # 不能用这个做注册
        else:
            y_train[i][0] = 1  # 不能用这个做注册
    y_train = y_train.reshape(1, -1)[0]
    y_train = y_train.astype(np.int)
    # 对 标签 独热编码 将存在数据类别的那一类用 1 表示，不存在用 0 表示
    y_train = tf.one_hot(y_train, N_classes)

    return (x_train, y_train)


def train_handler(src_path, username):
    x_train, y_train = read_csv(src_path, username)
    cnt = 0
    # 将测试集和训练集全部转化为numpy类型数据
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        y_train = y_train.eval(session=sess)
        x_train = np.array(x_train)
    for i in range(y_train.shape[0]):
        if int(y_train[i][0]) == 0:
            cnt += 1
    # print(cnt)
    #  待用户注册数据收集到10份的时候再做训练
    if cnt != 10:
        return '1'

    graph1 = tf.Graph()
    with graph1.as_default():
        x = tf.placeholder(tf.float64, [None, time_step * input_size])
        image = tf.reshape(x, [-1, time_step, input_size])
        y = tf.placeholder(tf.float64, [None, N_classes])
        rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=num_units)
        outputs, final_state = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=image, dtype=tf.float64)

        output = tf.layers.dense(inputs=outputs[:, -1, :], units=N_classes)

        loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=output)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        # 取出概率最高的值 并判断 y 跟 output 是否是同一下标
        # tf.argmax 返回一维向量中最大值的索引
        # tf.equal 返回一个布尔型的列表
        correct_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(output, axis=1))
        # tf.cast 把布尔型列表转换为 float32 ，如[T,T,F]，那么准确率为 66.6%
        # tf.reduce_mean 对一维向量取平均值
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

        saver = tf.train.Saver()
        # print(saver)

    with tf.Session(graph=graph1) as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(iterations):
            _, loss_ = sess.run([train_op, loss], {x: x_train, y: y_train})
        saver.save(sess, 'net/my_net.ckpt')

    return '1'


def test_handler(src_path, username):
    x_test, y_test = read_csv(src_path, username)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        y_test = y_test.eval(session=sess)

    graph1 = tf.Graph()
    # 重新定义了一遍网络，包括占位符之类，目前还没想到更好的方法
    with graph1.as_default():
        x = tf.placeholder(tf.float64, [None, time_step * input_size])
        image = tf.reshape(x, [-1, time_step, input_size])
        y = tf.placeholder(tf.float64, [None, N_classes])
        rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=num_units)
        outputs, final_state = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=image, dtype=tf.float64)
        # 做测试的时候，只需要知道 output
        output = tf.layers.dense(inputs=outputs[:, -1, :], units=N_classes)

        saver = tf.train.Saver()

    with tf.Session(graph=graph1) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, 'net/my_net.ckpt')
        y_pred = np.zeros((y_test.shape[0],))
        y_true = np.zeros((y_test.shape[0],))

        # print(x_test[0])
        # print(x_test[0].reshape(1, -1))
        for step in range(y_test.shape[0]):
            prediction = sess.run(output, {x: x_test[step].reshape(1, -1)})
            pred_one_hot = np.argmax(prediction, axis=1)
            y_pred[step] = pred_one_hot[0]
            # 450 * 1 -> 1 * 450
            y_true[step] = np.argmax(y_test[step].reshape(1, -1))

            print("预测用户为：", pred_one_hot[0], end=' ')
            print("实际用户为：", np.argmax(y_test[step].reshape(1, -1)))

            if pred_one_hot[0] == np.argmax(y_test[step].reshape(1, -1)):
                return '1'
            else:
                return '0'

# print("Recall", recall_score(y_true, y_pred))
# print("Precision", precision_score(y_true, y_pred))
# print("CMatrix\n", confusion_matrix(y_true, y_pred))


# train_handler(all_register, '7654321')
# test_handler(test_src_path, '7654321')
