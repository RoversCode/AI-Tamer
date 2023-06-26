#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   adaboost_ss.py
@Time    :   2023/05/31 11:37:56
@Author  :   ChengHee 
@Version :   1.0
@Contact :   1059885524@qq.com
@Desc    :   代码解析
在 __init__ 方法中初始化了算法的参数，包括迭代轮数和弱分类器权重缩减系数。
在 init_args 方法中初始化了样本集和标签集，以及一些其他成员变量。
_G 方法实现了单一特征阈值判定分类器，即弱分类器。它通过遍历所有可能的切分点，找到最小化误差的分类方式作为当前弱分类器。
_alpha 方法用于计算每个弱分类器的权重系数 alpha。
_Z 方法计算规范化因子，使用 alpha 和当前弱分类器对样本进行加权，然后将结果加和起来。
_w 方法用于更新样本的权重。
G 方法实现了 G(x) 的线性组合，用于将多个弱分类器组合成一个更强的分类器。
fit 方法用于训练模型，通过不断迭代寻找最优弱分类器，并记录下其相关信息。最后保存所有弱分类器和其对应的权重系数。
predict 方法用于预测新样本的分类结果，通过线性组合多个弱分类器的结果，得到最终的分类结果。
'''

# here put the import lib
import numpy as np


class AdaBoost:
    '''
    input:n_estimators(int):迭代轮数
          learning_rate(float):弱分类器权重缩减系数
    '''
    def __init__(self, n_estimators=50, learning_rate=1.0):
        self.clf_num = n_estimators
        self.learning_rate = learning_rate
    def init_args(self, datasets, labels):
        self.X = datasets
        self.Y = labels
        self.M, self.N = datasets.shape
        # 弱分类器数目和集合
        self.clf_sets = []
        # 初始化weights
        self.weights = [1.0/self.M]*self.M
        # G(x)系数 alpha
        self.alpha = []
    #********* Begin *********#
    def _G(self, features, labels, weights):
        '''
        input:features(ndarray):数据特征
              labels(ndarray):数据标签
              weights(ndarray):样本权重系数
        '''
        m = len(features)
        error = 100000.0 # 无穷大
        best_v = 0.0
        # 单维features
        features_min = min(features)
        features_max = max(features)
        # 步长?
        n_step = (features_max - features_min + self.learning_rate) // self.learning_rate
        direct, compare_array = None, None
        for i in range(1, int(n_step)):
            v = features_min + self.learning_rate * i
            if v not in features:
                # 误分类计算
                compare_array_positive = np.array([1 if features[k] > v else -1 for k in range(m)])
                weight_error_positive = sum([weights[k] for k in range(m) if compare_array_positive[k] != labels[k]])
                compare_array_nagetive = np.array([-1 if features[k] > v else 1 for k in range(m)])
                weight_error_nagetive = sum([weights[k] for k in range(m) if compare_array_nagetive[k] != labels[k]])
                if weight_error_positive < weight_error_nagetive:
                    weight_error = weight_error_positive
                    _compare_array = compare_array_positive
                    direct = 'positive'
                else:
                    weight_error = weight_error_nagetive
                    _compare_array = compare_array_nagetive
                    direct = 'nagetive'
                if weight_error < error:
                    error = weight_error
                    compare_array = _compare_array
                    best_v = v
        return best_v, direct, error, compare_array
    # 计算alpha
    def _alpha(self, error):
        return 0.5 * np.log((1-error)/error)
    # 规范化因子
    def _Z(self, weights, a, clf):
        return sum([weights[i]*np.exp(-1*a*self.Y[i]*clf[i]) for i in range(self.M)])
    # 权值更新
    def _w(self, a, clf, Z):
        for i in range(self.M):
            self.weights[i] = self.weights[i]*np.exp(-1*a*self.Y[i]*clf[i])/ Z
    # G(x)的线性组合
    def G(self, x, v, direct):
        if direct == 'positive':
            return 1 if x > v else -1 
        else:
            return -1 if x > v else 1 
    def fit(self, X, y):
        '''
        X(ndarray):训练数据
        y(ndarray):训练标签
        '''
        self.init_args(X, y)
        for epoch in range(self.clf_num):
            best_clf_error, best_v, clf_result = 100000, None, None
            # 根据特征维度, 选择误差最小的
            for j in range(self.N):
                features = self.X[:, j]
                # 分类阈值，分类误差，分类结果
                v, direct, error, compare_array = self._G(features, self.Y, self.weights)
                if error < best_clf_error:
                    best_clf_error = error
                    best_v = v
                    final_direct = direct
                    clf_result = compare_array
                    axis = j
                if best_clf_error == 0:
                    break
            # 计算G(x)系数a
            a = self._alpha(best_clf_error)
            self.alpha.append(a)
            # 记录分类器
            self.clf_sets.append((axis, best_v, final_direct))
            # 规范化因子
            Z = self._Z(self.weights, a, clf_result)
            # 权值更新
            self._w(a, clf_result, Z)
    def predict(self, data):
        '''
        input:data(ndarray):单个样本
        output:预测为正样本返回+1，负样本返回-1
        '''
        result = 0.0
        for i in range(len(self.clf_sets)):
            axis, clf_v, direct = self.clf_sets[i]
            f_input = data[axis]
            result += self.alpha[i] * self.G(f_input, clf_v, direct)
        return 1 if result > 0 else -1
    #********* End *********#

#加载数据
def load_data():
    from sklearn.datasets import load_iris
    import pandas as pd
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:, [0, 1, -1]])
    #将标签为0的数据标签改为-1
    for i in range(len(data)):
        if data[i,-1] == 0 or data[i,-1] == 2:
            data[i,-1] = -1
    return data[:,:2], data[:,-1]


if __name__ == '__main__':
    train_x, train_y = load_data()
    test_x = train_x[140:,:]
    test_y = train_y[140:]
    clf = AdaBoost(n_estimators=30, learning_rate=0.5)
    clf.fit(train_x, train_y)
    print(clf.predict(test_x[0]), test_y[0])
    print(clf.predict(test_x[1]), test_y[1])
    print(clf.predict(test_x[2]), test_y[2])
    print(clf.predict(test_x[3]), test_y[3])
    print(clf.predict(test_x[4]), test_y[4])
    print(clf.predict(test_x[5]), test_y[5])
    print(clf.predict(test_x[6]), test_y[6])
    print(clf.predict(test_x[7]), test_y[7])
    print(clf.predict(test_x[8]), test_y[8])
    print(clf.predict(test_x[9]), test_y[9])
