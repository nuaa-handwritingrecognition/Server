# -*- coding: utf-8 -*-
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import tqdm
from sklearn.metrics import precision_score, recall_score, confusion_matrix

# 参数
N_classes = 2  # 类别个数（二分类）
time_step = 150  # 一共有150个时间戳
input_size = 3  # 每个时间戳传入3个数据

# 超参数
learning_rate = 0.01  # 学习速率
num_units = 100  # 隐藏层神经元个数
iterations = 250  # 迭代次数
batch_size = 50  # 每次传入50个


# 读取csv文件
def read_csv():
    dataframe = pd.read_csv("./../../data/pre_collected_data.csv", header=None)
    x = dataframe.iloc[:, :450]  # 第0-449列
    y = dataframe.iloc[:, 450]  # 第450列
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.2)
    y_train = y_train.astype(np.int)
    y_test = y_test.astype(np.int)
    # 对 标签 独热编码 将存在数据类别的那一类用 1 表示，不存在用 0 表示
    y_train = tf.one_hot(y_train, N_classes)
    y_test = tf.one_hot(y_test, N_classes)

    return (x_train, x_test, y_train, y_test)


x_train, x_test, y_train, y_test = read_csv()

# 将测试集和训练集全部转化为numpy类型数据
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # print(type(y_train))
    # Tensor 转换成 numpy
    y_train = y_train.eval(session=sess)
    # print(type(y_train))
    y_test = y_test.eval(session=sess)

    x_train = np.array(x_train)
    x_test = np.array(x_test)

graph1 = tf.Graph()

with graph1.as_default():
    x = tf.placeholder(tf.float64, [None, time_step * input_size])
    # -1 指根据 x 来确定第一维的数据
    image = tf.reshape(x, [-1, time_step, input_size])
    y = tf.placeholder(tf.float64, [None, N_classes])
    # 建立神经网络（3*100*2）
    rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=num_units)
    outputs, final_state = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=image, dtype=tf.float64)

    # print(outputs) # ? * 150 * 100
    # print(outputs[:, -1, :])
    # 取最后一个时间戳的 outputs 并送入全连接层 得到 shape=(?, 2) 的 output
    output = tf.layers.dense(inputs=outputs[:, -1, :], units=N_classes)
    # print(output)

    # 用 softmax 求损失值
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=output)
    # 用 Loss 反向传播，AdamOptimizer 梯度下降
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # 取出概率最高的值 并判断 y 跟 output 是否是同一下标
    # tf.argmax 返回一维向量中最大值的索引
    # tf.equal 返回一个布尔型的列表
    correct_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(output, axis=1))
    # print(correct_prediction) # shape=(?,1)

    # tf.cast 把布尔型列表转换为 float32 ，如[T,T,F]，那么准确率为 66.6%
    # tf.reduce_mean 对一维向量取平均值
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    saver = tf.train.Saver()
    print(saver)

with tf.Session(graph=graph1) as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(iterations):
        _, loss_ = sess.run([train_op, loss], {x: x_train, y: y_train})
        if step % 10 == 0:
            predict = sess.run(output, {x: x_test[0].reshape(1, -1)})

            accuracy_ = sess.run(accuracy, {x: x_test, y: y_test})
            print("train loss:", loss_, " accuracy:", accuracy_, "pred", predict)
    saver.save(sess, 'net/my_net.ckpt')

with tf.Session(graph=graph1) as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, 'net/my_net.ckpt')
    y_pred = np.zeros((y_test.shape[0],))
    y_true = np.zeros((y_test.shape[0],))

    print(x_test[0])
    print(x_test[0].reshape(1, -1))
    for step in range(y_test.shape[0]):
        prediction = sess.run(output, {x: x_test[step].reshape(1, -1)})
        pred_one_hot = np.argmax(prediction, axis=1)
        y_pred[step] = pred_one_hot[0]
        # 450 * 1 -> 1 * 450
        y_true[step] = np.argmax(y_test[step].reshape(1, -1))
        print("第", step, "个数的预测结果为：", pred_one_hot[0], end=' ')
        print("实际值为：", np.argmax(y_test[step].reshape(1, -1)))

print("Recall", recall_score(y_true, y_pred))
print("Precision", precision_score(y_true, y_pred))
print("CMatrix\n", confusion_matrix(y_true, y_pred))