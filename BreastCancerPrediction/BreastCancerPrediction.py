# -*- coding:utf-8 -*-

import pickle
import numpy as np
import pandas as pd
from sklearn import svm, neighbors
from sklearn.model_selection import train_test_split

"""数据集说明：数据集来自威斯康星州医院的699条乳腺肿瘤数据，每条数据包含以下内容：
   #  Attribute                     Domain
   -- -----------------------------------------
   1. Sample code number            id number
   2. Clump Thickness               1 - 10
   3. Uniformity of Cell Size       1 - 10
   4. Uniformity of Cell Shape      1 - 10
   5. Marginal Adhesion             1 - 10
   6. Single Epithelial Cell Size   1 - 10
   7. Bare Nuclei                   1 - 10
   8. Bland Chromatin               1 - 10
   9. Normal Nucleoli               1 - 10
  10. Mitoses                       1 - 10
  11. Class:                        (2 for benign, 4 for malignant)
  
运用机器学习算法，通过对这些数据特征的处理来预测肿瘤是良性还是恶性。
"""

'''获取并预处理原始数据集'''
df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)

# 去除id列，其与肿瘤是否为良性还是恶性无关，加入会严重影响分类的结果。
df.drop(['id'], 1, inplace=True)

'''将数据集划分为训练集及测试集合'''
X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

'''选择算法、训练算法并测试算法'''
cls_dict = {
    'SVM-SVC': svm.SVC(),
    'KNN': neighbors.KNeighborsClassifier()
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

    # 测试算法
    print "%s Algorithm Accuracy: %s" % (name, cls.score(X_test, y_test))

    # 预测
    samples = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 1, 2, 2, 2, 3, 2, 1]])
    samples = samples.reshape(len(samples), -1)
    prediction = cls.predict(samples)
    print "%s Algorithm prediction: %s\n" % (name, prediction)
