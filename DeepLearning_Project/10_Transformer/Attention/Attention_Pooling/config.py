import torch

'''原始数据'''
n_train = 50 # 训练样本数

x_train, _ = torch.sort(torch.rand(n_train) * 5) # 源序列
def f(x):
    return 2 * torch.sin(x) + x**0.8
y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,)) # 目标序列

x_test = torch.arange(0, 5, 0.1) # 测试序列
y_truth = f(x_test) # 测试样本的真实输出

class Config:
    x_train = x_train
    y_train = y_train
    x_test = x_test
    y_truth = y_truth
    n_test = len(x_test) 