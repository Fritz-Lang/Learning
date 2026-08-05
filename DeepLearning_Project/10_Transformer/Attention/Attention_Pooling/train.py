import torch
import matplotlib.pyplot as plt
from config import Config
from utils import plot_kernel_reg
from model import fn1, fn2, fn3

'''平均汇聚'''
y_hat = fn1(Config.y_train.mean(), Config.n_test)
plot_kernel_reg(y_hat)
plt.show()

'''KNN'''
y_hat = fn2(Config.x_train, Config.y_train, Config.n_test, 15)
plot_kernel_reg(y_hat)
plt.show()

''' 非参数注意力汇聚'''
y_hat = fn3(Config.x_train, Config.y_train, Config.n_test, Config.x_test)
plot_kernel_reg(y_hat)
plt.show()