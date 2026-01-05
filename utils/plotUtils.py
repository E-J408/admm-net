import numpy as np
import matplotlib.pyplot as plt


def matlab_style_plots():
    """设置MATLAB风格的绘图"""
    # 设置中文
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
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


def plot_predictions_vs_truth(f, tau, res):
    """
    绘制预测值与真实值的对比图

    参数:
    f: 真实频率值数组，形状 (L,)
    tau: 真实延迟值数组，形状 (L,)
    res: 预测值列表，每个元素为 [tau_pred, f_pred, peak_value]
    """
    matlab_style_plots()
    L = len(f)

    # 检查输入维度
    assert len(tau) == L, "f 和 tau 的长度必须相同"
    assert len(res) == L, "res 的长度必须与 f 相同"

    # 准备数据
    f_truth = f
    tau_truth = tau

    # 从res中提取预测值
    f_pred = np.array([res[i][1] for i in range(L)])
    tau_pred = np.array([res[i][0] for i in range(L)])

    # 创建图形
    plt.figure(figsize=(10, 8))

    # 绘制真实值
    plt.scatter(tau_truth, f_truth, c='white', s=50, marker='o',
                label='真实值参数位置', alpha=0.7, edgecolors='black', linewidth=1.5)

    # 绘制预测值
    plt.scatter(tau_pred, f_pred, c='blue', s=40, marker='x',
                label='算法峰尖位置', alpha=0.7, linewidth=1.5)

    # # 添加从真实值到预测值的连线
    # for i in range(L):
    #     plt.plot([f_truth[i], f_pred[i]], [tau_truth[i], tau_pred[i]],
    #              'gray', alpha=0.3, linewidth=0.5)

    # 设置图形属性
    plt.ylabel('归一化多普勒频率/Hz', fontsize=12)
    plt.xlabel('归一化时延/s', fontsize=12)
    plt.title('本文算法峰尖过滤结果', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 添加坐标轴范围调整
    f_min = -0.5
    f_max = 0.5
    tau_min = 0
    tau_max = 1

    # 添加一些边距
    f_margin = (f_max - f_min) * 0.1
    tau_margin = (tau_max - tau_min) * 0.1

    plt.ylim(f_min - f_margin, f_max + f_margin)
    plt.xlim(tau_min - tau_margin, tau_max + tau_margin)

    plt.tight_layout()
    plt.show()



# 示例使用
if __name__ == "__main__":
    # 创建示例数据
    np.random.seed(42)
    L = 20

    # 真实值
    f_truth = np.random.uniform(0.1, 0.9, L)
    tau_truth = np.random.uniform(1, 10, L)

    # 预测值（添加一些随机误差）
    res = []
    for i in range(L):
        # 在真实值基础上添加随机误差
        f_pred = f_truth[i] + np.random.normal(0, 0.05)
        tau_pred = tau_truth[i] + np.random.normal(0, 0.5)
        peak_value = np.random.uniform(0.5, 1.0)  # 峰值大小，绘制中不使用

        res.append(np.array([tau_pred, f_pred, peak_value]))

    # 绘制图形
    plot_predictions_vs_truth(f_truth, tau_truth, res)