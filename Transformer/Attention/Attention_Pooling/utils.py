from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
from config import Config


def use_svg_display():  # @save
    """使用svg格式在Jupyter中显示绘图"""
    backend_inline.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):  # @save
    """设置matplotlib的图表大小"""
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

# @save


def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """绘制数据点"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else plt.gca()

    # 如果X有一个轴，输出True
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    # 通过数据在空白画布轴上画画，（x数据，y数据，线的样式）
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    # 将空白画布裱进画框里，设置画框的参数（画布，x轴名称，y轴名称，x轴的范围，y轴的范围，线的名称）
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


def plot_kernel_reg(y_hat):
    plot(Config.x_test, [Config.y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'],
         xlim=[0, 5], ylim=[-1, 5])
    plt.plot(Config.x_train, Config.y_train, 'o', alpha=0.5)


def plot_truth():
    plot(Config.x_test, Config.y_truth, 'x', 'y',
         legend=['Truth'], xlim=[0, 5], ylim=[-1, 5])
    plt.plot(Config.x_train, Config.y_train, 'o', alpha=0.5)
