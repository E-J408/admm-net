import numpy as np
import matplotlib.pyplot as plt


def matlab_style_plots():
    """设置MATLAB风格的绘图"""
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['figure.figsize'] = [8, 6]


def plot_comparison(y, Psi, b, e):
    """绘制结果对比图"""
    matlab_style_plots()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 原始信号实部
    axes[0, 0].plot(np.real(b + e))
    axes[0, 0].set_title('原始信号实部')
    axes[0, 0].set_xlabel('样本索引')
    axes[0, 0].set_ylabel('幅度')

    # 原始信号虚部
    axes[0, 1].plot(np.imag(b + e))
    axes[0, 1].set_title('原始信号虚部')
    axes[0, 1].set_xlabel('样本索引')
    axes[0, 1].set_ylabel('幅度')

    # 误差信号
    axes[1, 0].plot(np.abs(e))
    axes[1, 0].set_title('误差信号幅度')
    axes[1, 0].set_xlabel('样本索引')
    axes[1, 0].set_ylabel('|e|')

    # Psi的幅度
    axes[1, 1].plot(np.abs(Psi.flatten()))
    axes[1, 1].set_title('Psi幅度')
    axes[1, 1].set_xlabel('索引')
    axes[1, 1].set_ylabel('|Psi|')

    plt.tight_layout()
    plt.show()
