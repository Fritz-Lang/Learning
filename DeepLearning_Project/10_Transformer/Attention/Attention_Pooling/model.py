import torch
from torch import nn

'''平均汇聚：计算所有训练样本输出值的平均值作为预测值'''
def fn1(y_train, num_train):
    return torch.repeat_interleave(y_train.mean(), num_train)

'''KNN：取k个与x最接近的键，并将它们的值平均作为估计值'''
def fn2(x_train, y_train, num_train, k):
    y_knn = torch.zeros(num_train)
    for i, x in enumerate(x_train):
        distances = torch.square(x_train - x)
        nearest_indices = torch.argsort(distances)[:k]
        y_knn[i] = torch.mean(y_train[nearest_indices])
    return y_knn

'''Nadaraya–Watson核回归：考虑集合中的所有点，根据每个点与x的接近程度来加权其贡献'''
def fn3(x_train, y_train, num_train, x_test):
    x_repeat = x_test.repeat_interleave(num_train).reshape((-1, num_train))
    attention_weights = nn.functional.softmax(-(x_repeat - x_train)**2 / 2, dim=1)
    return torch.matmul(attention_weights, y_train)