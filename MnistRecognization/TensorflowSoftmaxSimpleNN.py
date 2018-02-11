# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

if __name__ == '__main__':
    hm_epochs = 20
    batch_size = 25

    mnist = read_data_sets("MNIST_data/", one_hot=True)

    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder("float", [None, 10])

    # 定义操作：softmax的交叉熵代价函数
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

    # 定义操作：选择梯度下降算法优化器，学习率为0.01
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cross_entropy)

    # 定义操作：获取预测准确的值，得tf.argmax返回最大的那个数值所在的下标。
    correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

    # 定义操作：初始化所有变量
    init = tf.global_variables_initializer()

    # 创建一个会话对象，启动图执行图中的所有操作。
    with tf.Session() as sess:
        # 执行操作：初始化所有变量
        sess.run(init)

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / float(batch_size))):
                # 获取大小为batch_size的小批量数据
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)

                # 执行操作：执行优化和交叉熵
                _, c = sess.run([optimizer, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})

                # 累积误差
                epoch_loss += c

            print "Epoch %d/%d, loss: %s" % (epoch + 1, hm_epochs, epoch_loss)

        # 执行操作：找出测试数据集中那些预测正确的标签。
        correct_predictions = sess.run(correct_predictions, feed_dict={x: mnist.test.images, y_: mnist.test.labels})

        # 确定正确预测项的比例，把布尔值转换成浮点数，然后取平均值。
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'))

        print "Accuracy: ", accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
