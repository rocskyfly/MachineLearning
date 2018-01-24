
```python
"""Reading a text-based dataset by pandas"""
# 读取以制表符分隔值的文本文件，定义第一列列名为type，余下所有的列为一整体，列名为message。
import pandas as pd

sms = pd.read_table('sms.tsv', header=None, names=['type', 'message'])

#查看sms Dataframe的前五行内容。
sms.head()
```


```python
# 统计type列中不同类型的总和
sms['type'].value_counts()
```


```python
# 将type列的内容进行数值转换，将结果存储在新的列label列中
sms['label'] = sms['type'].map({'ham': 0, 'spam': 1})

sms.head()
```


```python
# 分别获取原始数据集的特征和标签。
X=sms['message']
y=sms['label']
```


```python
# 随机将数据集划分成成70%训练集，30%测试集。
# 设置random_state参数：get the same output the first time you make the split.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.3)
```


```python
"""Vectorizing our dataset："""
# 初始化文本矢量类实例，将训练文本转化为稀疏矩阵
from sklearn.feature_extraction.text import CountVectorizer
cls = CountVectorizer()
X_train = cls.fit_transform(X_train)

# transform testing data (using fitted vocabulary) into a document-term matrix
X_test = cls.transform(X_test)
repr(X_test)
```


```python
"""Building and fit LogisticRegression model"""
# import and instantiate a logistic regression model
from sklearn.linear_model import LogisticRegression

cls = LogisticRegression()
cls.fit(X_train, y_train)
```


```python
"""Evaluate model"""
cls.score(X_test, y_test)
```


```python
"""偏斜类误差度量"""
from sklearn import metrics

# 获取混淆矩阵
print metrics.confusion_matrix(y_test, cls.predict(X_test))

# 获取F_1 Score
metrics.f1_score(y_test, cls.predict(X_test))
```
