#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   decision_sklearn.py
@Time    :   2023/05/30 11:10:03
@Author  :   ChengHee 
@Version :   1.0
@Contact :   1059885524@qq.com
@Desc    :   None
'''

# here put the import lib
import sklearn
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier



if __name__ == '__main__':
    iris = load_iris()
    train_df = iris['data']
    train_label = iris['target']
    test_df = train_df[0:10]
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(train_df[10:], train_label[10:])
    result = dt.predict(test_df)
    print(result)
    print(train_label[0:10])