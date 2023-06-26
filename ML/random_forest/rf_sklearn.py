#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   rf_sklearn.py
@Time    :   2023/05/30 12:01:26
@Author  :   ChengHee 
@Version :   1.0
@Contact :   1059885524@qq.com
@Desc    :   None
'''

# here put the import lib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits

def digit_predict(train_image, train_label, test_image):
    '''
    实现功能：训练模型并输出预测结果
    :param train_image: 包含多条训练样本的样本集，类型为ndarray,shape为[-1, 8, 8]
    :param train_label: 包含多条训练样本标签的标签集，类型为ndarray
    :param test_image: 包含多条测试样本的测试集，类型为ndarry
    :return: test_image对应的预测标签，类型为ndarray
    '''

    #************* Begin ************#
    # 训练集变形
    flat_train_image = train_image.reshape((-1, 64))
    # 测试集变形
    flat_test_image = test_image.reshape((-1, 64))

    rf = RandomForestClassifier(30, max_depth=20, random_state=42)
    rf.fit(flat_train_image, train_label)
    return rf.predict(flat_test_image)
    #************* End **************#

if __name__ == '__main__':
    data = load_digits()
    train_image = data['images']
    train_label = data['target']
    test_image = train_image[0:10]
    result = digit_predict(train_image[10:], train_label[10:], test_image)
    print(result)
    print(train_label[0:10])