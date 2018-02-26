# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets


def conv2d(xs, w):
    """卷积：使用1步长（stride size），使用左右补零填充边距操作，使得输入和输出的像素相同。"""
    return tf.nn.conv2d(input=xs, filter=w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(xs):
    """池化：采用最大值池化，ksize为池化窗口的大小，strides为移动的步长"""
    return tf.nn.max_pool(xs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """权重在初始化时应该加入少量的噪声来打破对称性以及避免0梯度。"""
    initial = tf.truncated_normal(shape=shape, stddev=0.1)

    return tf.Variable(initial)


def bias_variable(shape):
    """ReLu使用一个较小的正数来初始化偏置，以避免神经元输出恒为0"""
    initial = tf.constant(shape=shape, value=0.1)

    return tf.Variable(initial)


def convolution_neural_network(xs, n_classes=10):
    weights = {
        'w_conv1': weight_variable([5, 5, 1, 32]),  # 第一层卷积：使用5x5的卷积核，获得32个特征，输入的通道为1，输出通道为32。
        'w_conv2': weight_variable([5, 5, 32, 64]),  # 第二层卷积：使用5x5的卷积核，获得64个特征，输入的通道为32，输出通道为64。
        'w_fc': weight_variable([7 * 7 * 64, 1024]),  # 全连接层：输入图片的尺寸减小到7x7，将其输入至1024个神经元的全连接层。
        'w_out': weight_variable([1024, n_classes])
    }

    bias = {
        'b_conv1': bias_variable([32]),
        'b_conv2': bias_variable([64]),
        'b_fc': bias_variable([1024]),
        'b_out': bias_variable([n_classes])
    }

    # 将原始输入图片变成一个思维向量，第一维表示任意个输入，第二第三维分别对应图片的宽和高，
    # 最后一维代表图片输出通道数（颜色）。由于输入为灰度图所以输出通道数为1，若为rgb彩色图，则为3。
    xs = tf.reshape(xs, shape=[-1, 28, 28, 1])

    # 第一卷积层：卷积后加上偏置项，然后输入ReLU激活神经元。
    conv1 = tf.nn.relu(conv2d(xs, weights['w_conv1']) + bias['b_conv1'])
    pool1 = max_pool_2x2(conv1)  # 第一池化层
    conv2 = tf.nn.relu(conv2d(pool1, weights['w_conv2']) + bias['b_conv2'])  # 第二卷积层
    pool2 = max_pool_2x2(conv2)  # 第二池化层

    # 全连接层：将池化层输出的张量，转换成二维向量，第一维表示可以任意输入，第二维表示64个7x7的图片的一维向量。
    fc_input = tf.reshape(pool2, [-1, 7 * 7 * 64])
    fc_output = tf.nn.relu(tf.matmul(fc_input, weights['w_fc']) + bias['b_fc'])
    fc_output = tf.nn.dropout(fc_output, keep_prob)  # dropout：为了减少过拟合，以keep_prob的概率，随机丢弃一些神经元输出。

    # 输出层
    output = tf.matmul(fc_output, weights['w_out']) + bias['b_out']

    return output


def train_neural_network(xs, ys, n_classes=10, batch_size=128, hm_epochs=10, keep_rate=0.8):
    output = convolution_neural_network(xs, n_classes)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=ys))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y, keep_prob: keep_rate})
                epoch_loss += c

            print 'Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss

        correct = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print 'Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels, keep_prob: keep_rate})


if __name__ == '__main__':
    mnist = read_data_sets("MNIST_data/", one_hot=True)  # 准备训练的数据集

    x = tf.placeholder('float', [None, 784])
    y = tf.placeholder('float')
    keep_prob = tf.placeholder('float')

    train_neural_network(x, y, n_classes=10, batch_size=128, hm_epochs=10, keep_rate=0.8)
