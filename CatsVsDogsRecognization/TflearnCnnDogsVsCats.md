

```python
"""
数据集说明：数据集来自Kaggle猫狗识别比赛
训练集共有25000张已被标注的大小不尽相同的猫和狗的图片，猫狗图片数量各一半，图片命名规则为:[dog|cat].index.jpg
测试集共有12500张未被标注的大小不尽相同的猫和狗的图片。图片命名规则为:index.jpg

"""
```


```python
import os
import cv2
import numpy as np
from tqdm import tqdm
from random import shuffle
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
```


```python
#设置图片大小：长度和宽度
IMG_LENGTH = 50
IMG_WIDTH = 50
```


```python
def label_img(img):
    """
    获取训练集每张图片的标签。
    :param img: 输入为图片文件名
    :return: 输出为标签，猫为[1,0]，狗为[0,1]
    """
    word_label = img.split('.')[-3]

    if word_label == 'cat':
        return [1, 0]
    elif word_label == 'dog':
        return [0, 1]
    else:
        return [0, 0]
```


```python
def create_train_dataset(dest, img_length, img_width, train_dataset_name):
    """
    训练集数据预处理，获取训练集加标签的数据。
    :param dest: 训练集原始数据所在的目录
    :param img_length: 重新设置图片的长度
    :param img_width: 重新设置图片的宽度
    :param train_dataset_name: 保存处理后的数据集的名称
    :return: 训练集数据list，单个元素为[np.array(img), np.array(label)]
    """
    training_data = []
    for img in tqdm(os.listdir(dest)):  # tqdm进度条，用户只需要封装任意的迭代器tqdm(iterator)
        label = label_img(img)
        path = os.path.join(dest, img)

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_std = cv2.resize(img, (img_length, img_width))

        training_data.append([np.array(img_std), np.array(label)])

    # 将序列的所有元素随机排序
    shuffle(training_data)

    # 将training_data保存为文件，以备下次使用
    np.save(train_dataset_name, training_data)

    return training_data
```


```python
def create_test_dataset(dest, img_length, img_width, test_dataset_name):
    """
    测试集数据预处理，获取测试集数据。
    :param dest: 训练集原始数据所在的目录
    :param img_length: 重新设置图片的长度
    :param img_width: 重新设置图片的宽度
    :param test_dataset_name: 保存处理后的数据集的名称
    :return: 测试集数据list，单个元素为[np.array(img), img_num]
    """
    testing_data = []
    for img in tqdm(os.listdir(dest)):
        img_num = img.split('.')[0]
        path = os.path.join(dest, img)

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_std = cv2.resize(img, (img_length, img_width))

        testing_data.append([np.array(img_std), img_num])

    # 将testing_data保存为文件，以备下次使用
    np.save(test_dataset_name, testing_data)

    return testing_data
```


```python
def dataset_preprocessing(train_dir, test_dir, img_length=IMG_LENGTH, img_width=IMG_WIDTH):
    """
    获取并预处理训练集和测试集数据
    """
    train_dataset_name = 'train_dataset-{}x{}.npy'.format(img_length, img_width)
    test_dataset_name = 'test_dataset-{}x{}.npy'.format(img_length, img_width)

    if os.path.exists(train_dataset_name):
        train_dataset = np.load(train_dataset_name)
    else:
        train_dataset = create_train_dataset(train_dir, img_length, img_width, train_dataset_name)

    if os.path.exists(test_dataset_name):
        test_dataset = np.load(test_dataset_name)
    else:
        test_dataset = create_test_dataset(test_dir, img_length, img_width, test_dataset_name)

    return train_dataset, test_dataset

```


```python
class CNN(object):
    """
    定义卷积神经网络模型
    """

    def __init__(self, classifiers=2, lr=1e-3, img_length=IMG_LENGTH, img_width=IMG_WIDTH):
        self.classifiers = classifiers
        self.lr = lr
        self.img_length = img_length
        self.img_width = img_width
        self.log_dir = 'log'
        self.model_name = 'dogsvscats-{}-{}.model'.format(lr, '6conv-basic')

        tf.reset_default_graph()
        convnet = input_data(shape=[None, self.img_length, self.img_width, 1], name='input')

        convnet = conv_2d(convnet, nb_filter=32, filter_size=5, activation='relu')
        convnet = max_pool_2d(convnet, kernel_size=5)

        convnet = conv_2d(convnet, nb_filter=64, filter_size=5, activation='relu')
        convnet = max_pool_2d(convnet, kernel_size=5)

        convnet = conv_2d(convnet, nb_filter=32, filter_size=2, activation='relu')
        convnet = max_pool_2d(convnet, 2)

        convnet = conv_2d(convnet, nb_filter=64, filter_size=2, activation='relu')
        convnet = max_pool_2d(convnet, kernel_size=2)

        convnet = conv_2d(convnet, nb_filter=32, filter_size=2, activation='relu')
        convnet = max_pool_2d(convnet, kernel_size=2)

        convnet = conv_2d(convnet, nb_filter=64, filter_size=2, activation='relu')
        convnet = max_pool_2d(convnet, kernel_size=2)

        convnet = fully_connected(convnet, n_units=1024, activation='relu')

        convnet = dropout(convnet, keep_prob=0.5)

        convnet = fully_connected(convnet, classifiers, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=self.lr, loss='categorical_crossentropy',
                             name='targets')
        # convnet = regression(convnet, optimizer='adam', learning_rate=learn_rate, loss='binary_crossentropy',
        #                      name='targets')

        # 指定tensorboard_dir，可以将运行中生成的结构化数据放在此目录下，为tensorboard可视化提供数据,命令：
        # tensorboard --logdir full path of tensorboard_dir
        self.network = tflearn.DNN(convnet, tensorboard_dir=self.log_dir)

    def fit(self, train_dataset):
        # split out training and testing data，有标签数据分为训练集和测试集
        train = train_dataset[:-2500]
        test = train_dataset[-2500:]

        # separate my features and labels　特征，类别分离
        x = np.array([i[0] for i in train]).reshape(-1, self.img_length, self.img_width, 1)
        y = [i[1] for i in train]
        # print "x shape is:%s\t y size is:%s" % (x.shape, len(y))

        test_x = np.array([i[0] for i in test]).reshape(-1, self.img_length, self.img_width, 1)
        test_y = [i[1] for i in test]
        # print "test x shape is:%s\t test y size is:%s" % (test_x.shape, len(test_y))

        # run_id for tensorboard
        self.network.fit({'input': x}, {'targets': y}, n_epoch=5,
                         validation_set=({'input': test_x}, {'targets': test_y}),
                         snapshot_step=500, show_metric=True, run_id=self.model_name)

        return self.network

    def save(self):
        self.network.save(self.model_name)

        return True

    def load(self):
        if os.path.exists('{}.meta'.format(self.model_name)):
            self.network.load(self.model_name)
            print "model loaded!"
            return self.network
        else:
            print "no existed model, need training"
            return False

    def predict(self, dataset):
        if 0 < len(dataset) <= 12:
            fig = plt.figure()

            # predict first 12 data in test data
            for num, data in enumerate(dataset):  # enumerate将其组成一个索引序列，利用它可以同时获得索引和值
                img_data = data[0]
                img_index = data[1]

                # 将画布分割成3行4列，图像画在从左到右从上到下的第num+1块
                y = fig.add_subplot(3, 4, num + 1)
                orig = img_data
                data = img_data.reshape(self.img_length, self.img_width, 1)

                model_out = self.network.predict([data])[0]

                if np.argmax(model_out) == 1:
                    str_label = 'Dog-' + img_index
                else:
                    str_label = 'Cat-' + img_index

                y.imshow(orig, cmap='gray')
                plt.title(str_label)
                y.axes.get_xaxis().set_visible(False)
                y.axes.get_yaxis().set_visible(False)

            plt.show()
        else:
            with open('result.csv', 'w') as f:
                f.write('id,label\n')

            with open('result.csv', 'a') as f:
                for data in tqdm(dataset):
                    img_num = data[1]
                    img_data = data[0]
                    data = img_data.reshape(self.img_length, self.img_width, 1)
                    model_out = self.network.predict([data])[0]

                    f.write('{},{}\n'.format(img_num, model_out[1]))
            print "Result show in file result.csv"
```


```python
if __name__ == '__main__':
    train_dir_name = './train'
    test_dir_name = './test'

    # 获取训练数据集及测试数据集
    train_data, test_data = dataset_preprocessing(train_dir_name, test_dir_name)

    model = CNN()

    if not model.load():
        model.fit(train_data)
        model.save()

    model.predict(test_data)
```
