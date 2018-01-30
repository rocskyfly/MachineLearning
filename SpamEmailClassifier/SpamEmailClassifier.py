# -*-coding:utf-8-*-

import pickle
import pandas as pd
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

if __name__ == '__main__':
    """Reading a text-based dataset by pandas"""
    # 读取以制表符分隔值的文本文件，定义第一列列名为type，余下所有的列为一整体，列名为message。
    sms = pd.read_table('sms.tsv', header=None, names=['type', 'message'])

    # 统计type列中不同类型的总和
    print sms['type'].value_counts()

    # 将type列的内容进行数值转换，将结果存储在新的列label列中。
    sms['label'] = sms['type'].map({'ham': 0, 'spam': 1})

    # 分别获取原始数据集的特征和标签。
    X = sms['message']
    y = sms['label']

    # 设置random_state参数：get the same output the first time you make the split.
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.25)

    """Vectorizing our dataset"""
    # 初始化文本矢量类实例，训练文本并将其转化为稀疏矩阵。
    cls = CountVectorizer()
    X_train = cls.fit_transform(X_train)
    # print repr(X_train)

    # transform testing data (using fitted vocabulary) into a document-term matrix
    X_test = cls.transform(X_test)
    # print repr(X_test)

    """Building and fitting model"""
    # 采用逻辑回归和贝叶斯两种模型
    cls_dict = {
        'LogisticRegression': LogisticRegression(),
        'NaiveBayes': MultinomialNB()
    }
    # 训练并测试算法：若算法需要调优，可手动删除model序列化文件。
    for name, cls in cls_dict.items():
        try:
            with open('%s.pickle' % name, 'rb') as f:
                cls = pickle.load(f)
        except Exception, e:
            # 训练算法
            cls.fit(X_train, y_train)
            print e

            # 序列化算法
            with open('%s.pickle' % name, 'wb') as f:
                pickle.dump(cls, f)

        """Evaluating model"""
        # 获取混淆矩阵，及F_1 Score
        print "%s Algorithm Accuracy: %s\nconfusion matrix:%s\nF1_score:%s\n" \
              % (name, cls.score(X_test, y_test), metrics.confusion_matrix(y_test, cls.predict(X_test)),
                 metrics.f1_score(y_test, cls.predict(X_test)))
