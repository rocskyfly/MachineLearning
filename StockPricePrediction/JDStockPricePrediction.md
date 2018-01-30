

```python
#数据集说明：京东自2014年5月至2017年12月份公司股票交易数据共898条，通过这些历史数据建模，运用机器学习的方法，预测未来的股票价格。
```


```python
"""原始数据处理，并进行特征工程"""
import pandas as pd

# 获取原始数据，清洗并处理数据：将所有列中为空的或未知的数据用-99999替代.
df = pd.read_csv('JDHistoricalQuotes.csv')
df.replace('?', -99999, inplace=True)
df.fillna(-99999,inplace=True)

df.head()
```


```python
#将volume列转换为float类型；对日期进行升序排序；重新设置df的index为日期。
df['volume'] = df['volume'] * 1.0
df['date'] = pd.to_datetime(df['date'])
df.sort_values(by='date', inplace=True)
df.set_index(df['date'], inplace=True)

df.head()
```


```python
#获取数据的关键特征。
df['hl_pct']=(df['high']-df['close'])/df['close'] * 100.0
df['pct_change'] = (df['close']-df['open'])/df['open']*100.0
df =df[['close','hl_pct','pct_change', 'volume']]

df.head()
```


```python
# 取数据集的最后1%的close作为需要预测的数据，余下的作为训练集和测试数据集
import math
data_set_percent = 0.01
forecast_out = int(math.ceil(data_set_percent * len(df)))
df['label'] = df['close'].shift(-forecast_out)
df.tail(12)
```


```python
#获取数据集的特征，归一化数据集的特征。
import numpy as np
X = np.array(df.drop('label', 1))

from sklearn import preprocessing
X = preprocessing.scale(X)
```


```python
# 将数据集划分为训练集、测试集以及预测数据集。
X_validation = X[:-forecast_out]
X_predict = X[-forecast_out:]
df.dropna(inplace=True)
y=np.array(df['label'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_validation, y, test_size=0.2)

```


```python
"""训练模型并评估模型，做出预测"""

from sklearn.linear_model import LinearRegression
from sklearn import svm
import pickle

# 存放模型预测数据
forecast_dict = {}

# 分别定义四种模型：线性回归、SVM线性回归、SVM多项式回归、SVM高斯回归
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
```


```python
"""通过matplotlib图形化展示"""
# 获取所有可以使用的图形样式
from matplotlib import style
style.available
```


```python
# 获取要预测时间点的起点：即dataframe最后一行数据中的index的值（类型为：class pandas._libs.tslib.Timestamp）
last_date_object = df.iloc[-1].name
last_date_unix = last_date_object.value // 10 ** 9
one_day_seconds = 24 * 60 * 60
next_date_unix = last_date_unix + one_day_seconds
```


```python
for key, predict in forecast_dict.items():
    df['forecast-%s' % key ] = np.nan
    for value in predict:
        from datetime import datetime
        next_date = datetime.fromtimestamp(next_date_unix)
        next_date_unix += one_day_seconds
        df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [value]
    # 回到需要预测时间的起点。
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

import matplotlib.pyplot as plt

# legend 图例就是为了展示出每个数据对应的图像名称.
plt.legend()
# plot(x, y)不指定x默认为df的index
plt.plot(df['close'], color='red', label='Data')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Support Vector Regression')
plt.show()
```
