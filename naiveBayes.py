from __future__ import division
import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()  # 加载数据
y = iris.target
x = iris.data


def naive_bayes(x, y, predict):
    unique_y = list(set(y))
    #find the label number
    print (unique_y)
    label_num = len(unique_y)
    
    sample_num, dim = x.shape
    print (x.shape)

    joint_p = [1] * label_num
    print (joint_p)

    # 把所有的类别都过一遍，计算P(c)
    for (label_index, label) in enumerate(unique_y):
        # print (label_index, label)

        p_c = len(y[y == label]) / sample_num
#计算每个类别的p_c

        for (feature_index, x_i) in enumerate(predict):
            print (feature_index, x_i)
            tmp = x[y == label]
            #temp 是当前label在x中的取值
            joint_p[label_index] *= len([t for t in tmp[:, feature_index] if t == x_i]) / len(tmp)
            print (joint_p[label_index])
        
        joint_p[label_index] *= p_c

    tmp = joint_p[0]
    max_index = 0
    for (i, p) in enumerate(joint_p):

        if tmp < p:
            tmp = p
            max_index = i

    return unique_y[max_index]

# 测试所用的数据为数据集中最后一个数据，类别为2
out = naive_bayes(x, y, np.array([5.9, 3., 5.1, 1.8]))
