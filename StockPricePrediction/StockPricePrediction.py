# -*- coding:utf-8 -*-

import math
import pickle
import datetime
import pandas as pd
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    """原始数据处理，即特征工程"""
    # 获取原始数据，清洗并处理数据：将所有列中为空的或未知的数据用-99999替代
    df = pd.read_csv('JDHistoricalQuotes.csv')
    df.replace('?', -99999, inplace=True)
    df.fillna(-99999, inplace=True)

    # 将volume列转换为float类型；对日期进行升序排序；重新设置df的index为日期。
    df['volume'] = df['volume'] * 1.0
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)
    df.set_index(df['date'], inplace=True)

    # 获取数据的关键特征
    df['hl_pct'] = (df['high'] - df['close']) / df['close'] * 100.0
    df['pct_change'] = (df['close'] - df['open']) / df['open'] * 100.0
    df = df[['close', 'hl_pct', 'pct_change', 'volume', ]]

    # math.ceil向上取整，取最近(data_set_percent * len(df)天close作为要预测的数据。
    data_set_percent = 0.01
    forecast_out = int(math.ceil(data_set_percent * len(df)))
    df['label'] = df['close'].shift(-forecast_out)
    X = np.array(df.drop('label', 1))

    # TODO:which need to be fixed, if no scale dataset, the training will be hanged.
    X = preprocessing.scale(X)  # 归一化数据集

    X_validation = X[:-forecast_out]  # 将(1-data_set_percent)的数据作为训练集
    X_predict = X[-forecast_out:]  # 将data_set_percent的数据作为预测数据集
    df.dropna(inplace=True)
    y = np.array(df['label'])

    X_train, X_test, y_train, y_test = train_test_split(X_validation, y, test_size=0.2)  # 划分数据集：训练集和测试集

    """训练模型并评估模型，做出预测"""
    # 存放模型预测数据
    forecast_dict = {}

    cls_dict = {
        'LineRegression': LinearRegression(n_jobs=10),  # 设置并行数
        'SvmLinearRegression': svm.SVR(kernel='linear', C=1e3),
        'SvmPolyRegression': svm.SVR(kernel='poly', C=8, degree=3),
        'SvmRbfRegression': svm.SVR(kernel='rbf', C=1e3, gamma=0.1)
    }

    # 训练并测试算法：并序列化model，若算法需要调优，可手动删除model序列化文件。
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

        print "%s Algorithm Accuracy: %s" % (name, cls.score(X_test, y_test))

        forecast_dict.setdefault(name, (cls.predict(X_predict)))

    """通过matplotlib图形化展示"""
    # 获取所有可以使用图形的样式
    print style.available
    style.use('dark_background')

    # 获取要预测时间点的起点：即dataframe最后一行数据中的index的值（类型为：class pandas._libs.tslib.Timestamp）
    last_date_object = df.iloc[-1].name
    last_date_unix = last_date_object.value // 10 ** 9
    one_day_seconds = 24 * 60 * 60
    next_date_unix = last_date_unix + one_day_seconds

    for key, predict in forecast_dict.items():
        df['forecast-%s' % key] = np.nan
        for value in predict:
            next_date = datetime.datetime.fromtimestamp(next_date_unix)
            next_date_unix += one_day_seconds
            df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [value]  # 每行数据增加时间属性
        next_date_unix -= one_day_seconds * len(predict)

        if key == 'LineRegression':
            color = 'blue'
        elif key == 'SvmPolyRegression':
            color = 'yellow'
        elif key == 'SvmLinearRegression':
            color = 'green'
        else:
            color = 'white'
        df['forecast-%s' % key].plot(label=key, color=color)

    # legend 图例就是为了展示出每个数据对应的图像名称.
    plt.legend()
    # plot(x, y)不指定x默认为df的index
    plt.plot(df['close'], color='red', label='Data')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.show()
