
import numpy as np

#建议代码，也算是Begin-End中的一部分
from collections import  Counter
from sklearn.tree import DecisionTreeClassifier

class RandomForestClassifier():
    def __init__(self, n_model=10):
        '''
        初始化函数
        '''
        #分类器的数量，默认为10
        self.n_model = n_model
        #用于保存模型的列表，训练好分类器后将对象append进去即可
        self.models = []
        #用于保存决策树训练时随机选取的列的索引
        self.col_indexs = []
        


    def fit(self, feature, label):
        '''
        训练模型
        :param feature: 训练集数据，类型为ndarray
        :param label: 训练集标签，类型为ndarray
        :return: None
        '''

        #************* Begin ************#
        for i in range(self.n_model):
            m = len(feature)
            index = np.random.choice(m, m)
            col_index = np.random.permutation(len(feature[0]))[:int(np.log2(len(feature[0])))]
            sample_data = feature[index]
            sample_data = sample_data[:, col_index]
            sample_lable = label[index]
            model = DecisionTreeClassifier()
            model = model.fit(sample_data, sample_lable)
            self.models.append(model)
            self.col_indexs.append(col_index)
        #************* End **************#


    def predict(self, feature):
        '''
        :param feature:测试集数据，类型为ndarray
        :return:预测结果，类型为ndarray，如np.array([0, 1, 2, 2, 1, 0])
        '''
        #************* Begin ************#
        result = []
        vote = []
        for i, model in enumerate(self.models):
            f = feature[:, self.col_indexs[i]]
            r = model.predict(f)
            vote.append(r)
        vote = np.array(vote)
        for i in range(len(feature)):
            v = sorted(Counter(vote[:, i]).items(), key=lambda x: x[1], reverse=True)
            result.append(v[0][0])
        return np.array(result)
        #************* End **************#
