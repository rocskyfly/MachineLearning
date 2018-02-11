# -*- coding:utf-8 -*-
import random
import pickle
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets


def sigmoid(z):
    """实现S型函数方法"""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """# 实现sigmoid_prime，计算σ函数的导数"""
    return sigmoid(z) * (1 - sigmoid(z))


class QuadraticCost(object):
    """二次代价函数"""

    @staticmethod
    def fn(a, y):
        """返回代价函数的整体误差"""
        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(z, a, y):
        """返回二次代价函数输出层的误差：z为带权输入，a为激活值（实际输出值），y为实际值"""
        return (a - y) * sigmoid_prime(z)


class CrossEntropyCost(object):
    """交叉熵代价函数"""

    @staticmethod
    def fn(a, y):
        """回代价函数的整体误差，np.nan_to_num 确保将nan转换成0.0"""
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(z, a, y):
        """返回二次代价函数输出层的误差：z为带权输入，a为激活值（实际输出值），y为实际值"""
        return a - y


class Network(object):
    def __init__(self, sizes=list(), cost=CrossEntropyCost, eta=3.0, mini_batch_size=25, epochs=20, lmbda=0.02):
        # sizes为长度为3的列表，表示神经网络的层数为3。
        self.num_layers = len(sizes)

        # 若创建⼀个在第⼀层有2个神经元，第⼆层有3个神经元，最后层有1个神经元Network对象为：sizes=[2,3,1]
        self.sizes = sizes

        # 神经网络代价函数
        self.cost = cost

        # 随机初始化偏置：np.random.randn ⽣成均值为 0，标准差为1 的⾼斯分布。
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        # 随机初始化权重：⽣成均值为 0，标准差为1/sqrt(input) 的⾼斯分布；zip函数接受任意多个序列作为参数，返回一个tuple列表。
        self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(sizes[:-1], sizes[1:])]

        self.eta = eta  # eta代表学习率。
        self.mini_batch_size = mini_batch_size  # mini_batch_size代表每次迭代的数据数量。
        self.epochs = epochs  # epochs表示迭代次数。
        self.lmbda = lmbda  # lmbda为L2正则化参数.

    def fit(self, train_data, test_data=None):
        self.sgd(train_data, test_data)

    def predict(self, x_predictions):
        predictions = [np.argmax(self.feed_forward(x.reshape(784, 1))) for x in x_predictions]
        return predictions

    def score(self, test_data):
        """评估算法：通过前向传播算法获取测试数据集的网络的输出值，
        将输出值与测试数据集的标签进行比对，获取准确率"""
        n_test = len(test_data)
        test_results = [(np.argmax(self.feed_forward(x.reshape(784, 1))), np.argmax(y))
                        for (x, y) in test_data]

        return sum(int(x == y) for (x, y) in test_results) / float(n_test)

    def feed_forward(self, x):
        """前向传播算法：对每⼀层计算神经元的激活值：σ(wx + b)"""
        for b, w in zip(self.biases, self.weights):
            x = sigmoid(np.dot(w, x) + b)

        return x

    def sgd(self, train_data, test_data):
        """实现小批量随机梯度下降算法：train_data是训练数据集列表，每个元素为(x, y)；test_data用来评估
        每次小批量迭代后的准确率；"""
        n = len(train_data)
        for j in xrange(self.epochs):
            # 随机地将训练数据打乱，然后将它切分成每个大小为mini_batch_size的⼩批量数据集。
            random.shuffle(train_data)
            mini_batches = [train_data[k:(k + self.mini_batch_size)] for k in xrange(0, n,
                                                                                     self.mini_batch_size)]

            # 每⼀个 mini_batch应⽤⼀次梯度下降算法，根据单次梯度下降的迭代更新⽹络的权重和偏置。
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, n)

            if test_data:
                print "Epoch %s/%s accuracy: %s" % (j, self.epochs, self.score(test_data))

    def update_mini_batch(self, mini_batch, train_dataset_length):
        """对每个mini_batch应用一次梯度下降，更新网络权重和偏置"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]  # zeros返回来一个给定形状和类型的,用0填充的数组.
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # 对mini_batch中的每对训练数据应用反向传播算法，获取每对训练数据在每层上的代价函数的梯度和。
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.back_prop(x.reshape(784, 1), y.reshape(10, 1))
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # 应用小批量随机梯度下降算法，更新网络权重和偏置。
        self.weights = [(1 - self.eta * self.lmbda / train_dataset_length) * w - (self.eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (self.eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def back_prop(self, x, y):
        """反向传播算法：逐层获取代价函数关于权重和偏置的偏导数。"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # 前向传播：计算并保存每层神经元的的带权输入以及激活值。
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # 反向传播：首先计算并保存最后输出层的误差（偏置偏导数）以及权重偏导数。
        delta = self.cost.delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # 反向传播：自倒数第二层开始，逐层计算并保存输出误差（偏置偏导数）以及权重偏导数。
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sigmoid_prime(z)
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return nabla_b, nabla_w

    def load(self, filename='NeuralNetwork.pickle'):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.weights = data["weights"]
            self.biases = data["biaes"]
            self.sizes = data["sizes"]
            self.cost = data["cost"]
            self.eta = data["eta"]
            self.mini_batch_size = data["mini_batch_size"]
            self.epochs = data["epochs"]
        print "Load model successfully!"

    def save(self, filename='NeuralNetwork.pickle'):
        data = {
            "weights": self.weights,
            "biaes": self.biases,
            "sizes": self.sizes,
            "cost": self.cost,
            "eta": self.eta,
            "mini_batch_size": self.mini_batch_size,
            "epochs": self.epochs,
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print "Save mode successfully!"


if __name__ == '__main__':
    """数据集说明：minist 训练图片数据集总有55000条，测试图片数据集共有10000条"""
    # 读取mnist数据集
    mnist = read_data_sets("MNIST_data/", one_hot=True)

    # zip函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
    train_data_set = zip(mnist.train.images, mnist.train.labels)
    test_data_set = zip(mnist.test.images, mnist.test.labels)

    cls = Network([784, 400, 10])

    cls.fit(train_data_set, test_data_set)

    cls.save()

    print "Accuracy: %s" % cls.score(test_data_set)
