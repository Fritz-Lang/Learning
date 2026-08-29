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

'''Nadaraya–Watson核回归：考虑集合中的所有点，根据每个点与x的接近程度来加权其贡献（采用高斯核）'''
def fn3(x_train, y_train, num_train, x_test):
    x_repeat = x_test.repeat_interleave(num_train).reshape((-1, num_train))
    attention_weights = nn.functional.softmax(-(x_repeat - x_train)**2 / 2, dim=1)
    return torch.matmul(attention_weights, y_train)

'''带参数的注意力汇聚（高斯核）'''
class NWKernelRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))

    def forward(self, queries, keys, values):
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        self.attention_weights = nn.functional.softmax(-((queries - keys) * self.w)**2 / 2, dim=1)
        return torch.bmm(self.attention_weights.unsqueeze(1),values.unsqueeze(-1)).reshape(-1)