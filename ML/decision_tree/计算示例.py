#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   decisin_tree.py
@Time    :   2023/05/27 20:46:10
@Author  :   ChengHee 
@Version :   1.0
@Contact :   1059885524@qq.com
@Desc    :   该脚本是计算信息增益、信息增益率、基尼指数的示例
'''

# here put the import lib
import sklearn
import numpy as np
import math

'''
数据集
编号	性别	活跃度	是否流失
1	男	高	0
2	女	中	0
3	男	低	1
4	女	高	0
5	男	高	0
6	男	中	0
7	男	中	1
8	女	中	0
9	女	低	1
10	女	中	0
11	女	高	0
12	男	低	1
13	女	低	1
14	男	高	0
15	男	高	0
'''

def calcInfoGain():
    '''
    计算性别和活跃度信息增益
    '''
    # 性别信息增益
    d = -(10/15) * np.log2(10/15) - (5/15) * np.log2(5/15) # 总熵
    d_male = -(5/8) * np.log2(5/8) - (3/8) * np.log2(3/8) 
    d_female = -(5/7) * np.log2(5/7) - (2/7) * np.log2(2/7) 
    g1 = d - (8/15) * d_male - (7/15) * d_female # 性别信息增益
    print('性别信息增益：', g1)
    # 活跃度信息增益
    d_low = -(4/4) * np.log2(4/4) - 0 
    d_middle = -(1/5) * np.log2(1/5) - (4/5) * np.log2(4/5)
    d_high = -1 * np.log2(1) - 0
    g2 = d - (4/15) * d_low - (5/15) * d_middle - (6/15) * d_high # 活跃度信息增益
    print('活跃度信息增益：', g2)


def calcInfGainRatio():
    '''
    计算性别和活跃度的信息增益率
    '''
    # 性别信息增益
    d = -(10/15) * np.log2(10/15) - (5/15) * np.log2(5/15) # 总熵
    d_male = -(5/8) * np.log2(5/8) - (3/8) * np.log2(3/8) 
    d_female = -(5/7) * np.log2(5/7) - (2/7) * np.log2(2/7) 
    g1 = d - (8/15) * d_male - (7/15) * d_female # 性别信息增益
    # 性别信息增益率
    iv1 = - (8/15) * np.log2(8/15) - (7/15) * np.log2(7/15)
    g1_ratio = g1 / iv1
    print('性别信息增益率：', g1_ratio)
    # 活跃度信息增益
    d_low = -(4/4) * np.log2(4/4) - 0
    d_middle = -(1/5) * np.log2(1/5) - (4/5) * np.log2(4/5)
    d_high = -1 * np.log2(1) - 0
    g2 = d - (4/15) * d_low - (5/15) * d_middle - (6/15) * d_high # 活跃度信息增益
    # 活跃度信息增益率
    iv2 = - (4/15) * np.log2(4/15) - (5/15) * np.log2(5/15) - (6/15) * np.log2(6/15)
    g2_ratio = g2 / iv2
    print('活跃度信息增益率：', g2_ratio)


def calcGini():
    '''
    计算性别和活跃度的基尼指数
    '''
    # 性别基尼指数
    g_male = 1 - (5/8) ** 2 - (3/8) ** 2
    g_female = 1 - (5/7) ** 2 - (2/7) ** 2
    g1 = (8/15)*g_male + (7/15)*g_female
    print('性别基尼指数：', g1)
    # 活跃度基尼指数
    g_low = 1 - (4/4) ** 2
    g_middle = 1 - (1/5) ** 2 - (4/5) ** 2
    g_high = 1 - (6/6) ** 2
    g2 = (4/15)*g_low + (5/15)*g_middle + (6/15)*g_high
    print('活跃度基尼指数：', g2)


if __name__ == '__main__':
    calcInfoGain()  
    print('------------------'*3)
    calcInfGainRatio()
    print('------------------'*3)
    calcGini()