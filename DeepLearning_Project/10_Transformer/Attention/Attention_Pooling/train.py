import torch
from torch import nn
import matplotlib.pyplot as plt
from config import Config
from utils import plot_kernel_reg
from model import fn1, fn2, fn3, NWKernelRegression

#'''平均汇聚'''
# y_hat = fn1(Config.y_train.mean(), Config.n_test)
# plot_kernel_reg(y_hat)
# plt.show()

# '''KNN'''
# y_hat = fn2(Config.x_train, Config.y_train, Config.n_test, 15)
# plot_kernel_reg(y_hat)
# plt.show()

# ''' 非参数注意力汇聚'''
# y_hat = fn3(Config.x_train, Config.y_train, Config.n_test, Config.x_test)
# plot_kernel_reg(y_hat)
# plt.show()

'''带参数的注意力汇聚'''
# X_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输入
X_tile = Config.x_train.repeat((Config.n_train, 1))
# Y_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输出
Y_tile = Config.y_train.repeat((Config.n_train, 1))
# keys的形状:('n_train'，'n_train'-1)
keys = X_tile[(1 - torch.eye(Config.n_train)).type(torch.bool)].reshape((Config.n_train, -1))
# values的形状:('n_train'，'n_train'-1)
values = Y_tile[(1 - torch.eye(Config.n_train)).type(torch.bool)].reshape((Config.n_train, -1))

net = NWKernelRegression()
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.5)

for epoch in range(5):
    trainer.zero_grad()
    l = loss(net(Config.x_train, keys, values), Config.y_train)
    l.sum().backward()
    trainer.step()
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')

# keys的形状:(n_test，n_train)，每一行包含着相同的训练输入（例如，相同的键）
keys = Config.x_train.repeat((Config.n_test, 1))
# value的形状:(n_test，n_train)
values = Config.y_train.repeat((Config.n_test, 1))
y_hat = net(Config.x_test, keys, values).unsqueeze(1).detach()
plot_kernel_reg(y_hat)
plt.show()