# -*-coding:utf-8-*-
import pickle
from collections import Counter
from sklearn import metrics
from sklearn import datasets
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


class KNN:
    def __init__(self):
        self.x_train = None
        self.y_train = None

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test, k=3):
        predictions = []
        for element in x_test:
            # 无投票，简单找到离训练集最近的点，返回其label最为预测x的label
            label = self.closet(element)

            # 投票法
            # label = self.vote(element, k)

            predictions.append(label)

        return predictions

    def score(self, x_test, y_test):
        return metrics.accuracy_score(y_test, self.predict(x_test))

    def closet(self, element):
        def enc(a, b):
            return distance.euclidean(a, b)

        best_dist = enc(element, self.x_train[0])
        best_index = 0

        for index in range(1, len(self.x_train)):
            dist = enc(element, self.x_train[index])
            if dist < best_dist:
                best_dist = dist
                best_index = index

        return self.y_train[best_index]

    def vote(self, element, k):
        def enc(a, b):
            return distance.euclidean(a, b)

        k_list = []
        for index in range(k):
            best_dist = enc(element, self.x_train[index])
            k_list.append([index, best_dist])

        for index in range(k, len(self.x_train)):
            dist = enc(element, self.x_train[index])
            for i in range(k):
                if dist < k_list[i][1]:
                    k_list.pop(i)
                    k_list.insert(i, [index, dist])

        index_list = []
        for index in range(k):
            index_list.append(k_list[index][0])

        # list with one element, it's a tuple (index, times)
        counter = Counter(index_list)
        index = counter.most_common(1)[0][0]

        return self.y_train[index]


if __name__ == '__main__':
    # 获取鸢尾属植物数据集
    # This data sets consists of 3 different types of irises’ (Setosa, Versicolour, and Virginica)
    # petal and sepal length, stored in a 150x4 numpy.ndarray.The rows being the samples and
    # the columns being: Sepal Length,Sepal Width, Petal Length and Petal Width.'''
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target

    # 获取数据集大小
    print len(X)

    # 随机将数据集划分成成70%训练集，30%测试集。
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

    # 选择算法：选用自己实现的KNN
    cls_dict = {'Sklearn-KNN': KNeighborsClassifier(),
                'Rocky-KNN': KNN(),
                'DecisionTree': DecisionTreeClassifier(),
                }

    for name, cls in cls_dict.items():
        # 训练算法：并序列化，若算法需要调优，可手动删除序列化文件
        try:
            with open('%s.pickle' % name, 'rb') as f:
                cls = pickle.load(f)
        except Exception, e:
            # 训练算法
            cls.fit(X_train, Y_train)
            print e

            # 序列化算法
            with open('%s.pickle' % name, 'wb') as f:
                pickle.dump(cls, f)

        # 测试算法
        print "%s Algorithm Accuracy: %s" % (name, cls.score(X_test, Y_test))
