# -*-coding:utf-8-*-
# !/usr/bin/python
import pickle
from sklearn import tree
import graphviz  # Graphviz是图形绘制工具,可以很方便的用来绘制结构化的图形网络,支持多种格式输出

if __name__ == '__main__':
    # 数据说明：数据集为12组成人身高、体重、鞋码的组合数据，以及是男性还是女性"""
    # [height, weight, shoe size]
    X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37],
         [166, 65, 40], [190, 90, 47], [175, 64, 39], [177, 70, 40],
         [159, 55, 38], [171, 75, 42], [181, 85, 43], [148, 70, 42]]

    Y = ['male', 'female', 'female', 'female',
         'female', 'female', 'female', 'female',
         'male', 'female', 'male', 'male']

    try:
        with open('DecisionTree.pickle', 'rb') as f:
            clf = pickle.load(f)
    except Exception, e:
        # 训练算法
        clf = tree.DecisionTreeClassifier()
        clf.fit(X, Y)

        # 序列化算法
        with open('DecisionTree.pickle', 'wb') as f:
            pickle.dump(clf, f)

    # Visualization: graphviz export of the above tree trained on the entire dataset;
    # the results are saved in an output file GenderClassifier.pdf
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=['height', 'weight', 'shoe size'],
                                    class_names='gender',
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render('GenderClassifier')

    # make a prediction.
    prediction = clf.predict([[190, 70, 43], [156, 60, 36]])
    print prediction
