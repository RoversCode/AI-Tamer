#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   adaboost_sk.py
@Time    :   2023/05/30 17:22:06
@Author  :   ChengHee 
@Version :   1.0
@Contact :   1059885524@qq.com
@Desc    :   None
'''

# here put the import lib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier


def ada_classifier(train_data,train_label,test_data):
    '''
    input:train_data(ndarray):训练数据
          train_label(ndarray):训练标签
          test_data(ndarray):测试标签
    output:predict(ndarray):预测结果
    '''
    #********* Begin *********#
    ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier
           (max_depth=2, min_samples_split=10, min_samples_leaf=5),
            n_estimators=50,learning_rate=0.2)
    
    ada.fit(train_data,train_label)
    predict = ada.predict(test_data)
    #********* End *********# 
    return predict






























