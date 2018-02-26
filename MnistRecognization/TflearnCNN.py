# -*-coding:utf-8-*-

import os
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist

if __name__ == '__main__':
    x, y, test_x, test_y = mnist.load_data(one_hot=True)
    x = x.reshape([-1, 28, 28, 1])
    test_x = test_x.reshape([-1, 28, 28, 1])

    input_data = input_data(shape=[None, 28, 28, 1], name='input')

    # nb_filter表示输出通道数，filter_size表示卷积核大小5x5
    conv1 = conv_2d(input_data, nb_filter=32, filter_size=5, activation='relu')

    # kernel_size表示池化窗口大小2x2
    pool1 = max_pool_2d(conv1, kernel_size=2)

    conv2 = conv_2d(pool1, nb_filter=64, filter_size=5, activation='relu')

    pool2 = max_pool_2d(conv2, kernel_size=2)

    fc = fully_connected(pool2, n_units=1024, activation='relu')
    fc = dropout(fc, 0.8)

    output = fully_connected(fc, n_units=10, activation='softmax')

    network = regression(output, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(network, tensorboard_dir='log')

    if os.path.exists('tflearncnn.model.index') and os.path.exists('tflearncnn.model.meta'):
        model.load('tflearncnn.model')
    else:
        model.fit({'input': x}, {'targets': y}, n_epoch=10,
                  validation_set=({'input': test_x}, {'targets': test_y}),
                  snapshot_step=500, batch_size=128, show_metric=True, run_id='mnist')
        model.save('tflearncnn.model')

    # 做出预测
    print model.predict([test_x[1]])
