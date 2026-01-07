from utils.mathUtils import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import matplotlib.colors as colors
from skimage.morphology import local_maxima


def peak_search_func(phi, x, x_base, y, y_base):
    """
    计算二维空间中的峰值搜索函数值

    该函数通过构造特定的基函数向量并与其共轭进行Kronecker积运算，
    然后与输入信号进行内积计算，最终返回功率谱密度值。

    参数:
        phi: 复数数组，输入信号向量
        x: 数值，x方向的缩放因子
        x_base: 整数，x方向的基底维度
        y: 数值，y方向的缩放因子
        y_base: 整数，y方向的基底维度

    返回值:
        z: 数值，计算得到的功率谱密度值
    """
    # 构造y方向和x方向的Vandermonde向量
    s = vander_vec(0, (y_base - 1) * y, y_base)
    d = vander_vec(0, (x_base - 1) * x, x_base)
    # 计算s和d共轭的Kronecker积，构造二维基函数
    a = np.kron(s, np.conj(d))
    # 计算phi与基函数a的内积，取模平方得到功率谱密度
    z = np.abs(np.dot(phi.conj().T, a)) ** 2
    return z



def peak_search(phi, X, x_base, Y, y_base):
    """
    对二维数组中的每个元素执行峰值搜索操作

    参数:
        phi: 相位参数，传递给峰值搜索函数
        X: 二维数组，包含x方向的输入数据
        x_base: x方向的基础值，用于峰值搜索计算
        Y: 二维数组，包含y方向的输入数据
        y_base: y方向的基础值，用于峰值搜索计算

    返回:
        array_reg: 与X形状相同的复数数组，包含每个位置的峰值搜索结果
    """
    # 初始化结果数组，数据类型为复数
    array_reg = np.zeros((Y.shape[0], X.shape[1]))

    # 遍历二维数组的每个位置
    for idx1 in range(Y.shape[0]):
        for idx2 in range(X.shape[1]):
            # 对每个位置调用峰值搜索函数，将结果存储在对应位置
            array_reg[idx1, idx2] = peak_search_func(phi, X[idx1, idx2], x_base,
                                                     Y[idx1, idx2], y_base)
    return array_reg


def alt_peak_search(func_opts, opts=None):
    """
    在给定的参数空间中进行峰值搜索，并通过迭代细化找到更精确的峰值位置与强度。

    参数:
        func_opts (dict): 包含用于计算峰值的目标函数所需的基础参数。
            - 'phi': 目标函数中的相位参数。
            - 'xbase': X方向基础偏移量。
            - 'ybase': Y方向基础偏移量。
        opts (dict, 可选): 用户自定义的搜索选项，将覆盖默认设置。支持以下键：
            - 'xmin', 'xmax', 'xstep': X轴搜索范围及步长。
            - 'ymin', 'ymax', 'ystep': Y轴搜索范围及步长。
            - 'reducefactor': 每次迭代时缩小搜索步长的比例因子。
            - 'iter': 迭代次数，用于逐步提高精度。

    返回:
        numpy.ndarray: 形状为 (N, 3) 的数组，每行为一个检测到的峰值信息，
                       分别表示 [x坐标, y坐标, 峰值高度]。
    """

    # 默认参数
    default_opts = {
        'xmin': 0, 'xmax': 1, 'xstep': 0.01,
        'ymin': -0.5, 'ymax': 0.5, 'ystep': 0.01,
        'reducefactor': 0.1, 'iter': 1
    }

    # 合并用户参数
    if opts is None:
        opts = {}
    search_opts = {**default_opts, **opts}

    # 提取参数
    phi = func_opts['phi']
    x_base = func_opts['xbase']
    y_base = func_opts['ybase']

    xmin, xmax, xstep = search_opts['xmin'], search_opts['xmax'], search_opts['xstep']
    ymin, ymax, ystep = search_opts['ymin'], search_opts['ymax'], search_opts['ystep']
    reduce_factor, max_iter = search_opts['reducefactor'], search_opts['iter']

    # 生成初始搜索网格
    axis_x = np.arange(xmin, xmax-xstep, xstep)
    axis_y = np.arange(ymin, ymax-xstep, ystep)

    # 检查网格是否为空
    if len(axis_x) == 0 or len(axis_y) == 0:
        return np.zeros((0, 3))

    axis_X, axis_Y = np.meshgrid(axis_x, axis_y)

    # 计算整个搜索区域上的目标函数响应面
    axis_Z = peak_search(phi, axis_X, x_base, axis_Y, y_base)

    # 寻找所有局部极大值点的位置
    regional_maxima = local_maxima(axis_Z, connectivity=2)
    x_pos, y_pos = np.where(regional_maxima)
    num_peaks = len(x_pos)

    # 将粗略估计的峰值位置保存至结果矩阵前两列
    coarse_result = np.zeros((num_peaks, 2))
    for i in range(num_peaks):
        coarse_result[i, 0] = axis_X[x_pos[i], y_pos[i]]  # x位置
        coarse_result[i, 1] = axis_Y[x_pos[i], y_pos[i]]  # y位置

    # 初始化精细搜索的结果存储结构（增加一列用于记录峰值大小）
    refined_result = np.zeros((num_peaks, 3))
    refined_result[:, :2] = coarse_result.copy()

    local_xstep = xstep
    local_ystep = ystep

    # 多轮迭代优化峰值定位精度
    for j in range(max_iter):
        local_xstep = reduce_factor * local_xstep
        local_ystep = reduce_factor * local_ystep

        for k in range(num_peaks):
            # 定义当前峰值周围的局部搜索窗口
            local_xmin = max(xmin, refined_result[k, 0] - local_xstep)
            local_xmax = min(xmax - local_xstep, refined_result[k, 0] + local_xstep)
            local_ymin = max(ymin, refined_result[k, 1] - local_ystep)
            local_ymax = min(ymax - local_ystep, refined_result[k, 1] + local_ystep)

            # 若窗口无效则跳过该峰值处理
            if local_xmin >= local_xmax or local_ymin >= local_ymax:
                continue

            local_x = np.arange(local_xmin, local_xmax, local_xstep)
            local_y = np.arange(local_ymin, local_ymax, local_ystep)

            # 跳过空区间情况
            if len(local_x) == 0 or len(local_y) == 0:
                continue

            local_X, local_Y = np.meshgrid(local_x, local_y)
            local_Z = peak_search(phi, local_X, x_base, local_Y, y_base)

            # 修正
            local_Z_max = np.max(local_Z)
            # 创建掩码，只保留最大值位置
            local_Z_mask = (local_Z == local_Z_max)
            max_positions = np.where(local_Z_mask)
            if len(max_positions[0]) > 0:
                # 获取最大值位置
                max_pos = (max_positions[0][0], max_positions[1][0])
                refined_result[k, 0] = local_X[max_pos]
                refined_result[k, 1] = local_Y[max_pos]
                refined_result[k, 2] = local_Z_max

    return refined_result


def find_regional_maxima(image, footprint=None):
    """
    寻找区域极大值，替代MATLAB的imregionalmax
    """
    if footprint is None:
        footprint = np.ones((3, 3))

    if np.iscomplexobj(image):
        image = np.abs(image)
    # 使用最大滤波器寻找局部极大值
    image_max = ndimage.maximum_filter(image, footprint=footprint)

    # 局部极大值点是那些等于滤波后图像值的点（且不是背景）
    regional_max = (image == image_max)

    # # 去除边缘效应（可选）
    # regional_max = regional_max & (image > 0)  # 只保留正值区域

    return regional_max

#


def plot_peaks(func_opts, ground_truth, opts=None):
    """
    PLOTPEAKS的Python实现 - 寻峰表面作图

    参数:
    -----------
    func_opts : dict
        - phi: 寻峰中对应的估计phi向量
        - xbase: x轴方向的基础频率
        - ybase: y轴方向的基础频率

    ground_truth : dict
        - tau: 真实的tau向量
        - f: 真实的f向量

    opts : dict, optional
        绘图参数
    """

    # 默认参数
    default_opts = {
        'xmin': 0, 'xmax': 1, 'xstep': 0.01,
        'ymin': -0.5, 'ymax': 0.5, 'ystep': 0.01,
        'custom_title': 'Proposed method',
        'custom_legend': 'Predicted location of peaks'
    }

    # 合并参数
    if opts is None:
        opts = {}
    plot_opts = {**default_opts, **opts}

    # 提取参数
    phi = func_opts['phi']
    x_base = func_opts['xbase']
    y_base = func_opts['ybase']

    tau = ground_truth['tau']
    f = ground_truth['f']

    xmin, xmax, xstep = plot_opts['xmin'], plot_opts['xmax'], plot_opts['xstep']
    ymin, ymax, ystep = plot_opts['ymin'], plot_opts['ymax'], plot_opts['ystep']
    custom_title, custom_legend = plot_opts['custom_title'], plot_opts['custom_legend']

    # 生成网格
    axis_x = np.arange(xmin, xmax, xstep)
    axis_y = np.arange(ymin, ymax, ystep)
    axis_X, axis_Y = np.meshgrid(axis_x, axis_y)

    # 计算峰值搜索表面
    axis_Z = peak_search(phi, axis_X, x_base, axis_Y, y_base)

    # 生成真实峰值位置矩阵
    axis_Z_truth = ground_truth_matrix(tau, f, axis_X, axis_Y, xstep, ystep)

    axis_Z_max = np.max(axis_Z)
    axis_Z_peak = axis_Z * local_maxima(axis_Z, connectivity=2)

    # 创建图形
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制表面
    surf = ax.plot_surface(axis_X, axis_Y, np.abs(axis_Z),
                           cmap='viridis', alpha=0.8, linewidth=0,
                           antialiased=True)

    # 自定义颜色映射（类似MATLAB中的颜色）
    custom_colors = [
        [8, 67, 132],
        [19, 115, 178],
        [66, 166, 203],
        [119, 202, 197],
        [178, 225, 185],
        [214, 239, 208],
        [244, 250, 237]
    ]
    custom_colors = np.array(custom_colors) / 255.0

    # 创建自定义colormap
    cmap = colors.LinearSegmentedColormap.from_list('custom_cmap', custom_colors, N=256)
    surf.set_cmap(cmap)

    # 绘制预测峰值位置
    peak_positions = np.where(axis_Z_peak > 0)
    if len(peak_positions[0]) > 0:
        peak_x = axis_X[peak_positions]
        peak_y = axis_Y[peak_positions]
        peak_z = 2 * axis_Z_peak[peak_positions]  # 适当抬高标记

        ax.scatter(peak_x, peak_y, peak_z, c='blue', marker='*', s=50, alpha=0.3,
                   label=custom_legend, depthshade=False)


    # 绘制真实峰值位置
    truth_positions = np.where(axis_Z_truth > 0)
    if len(truth_positions[0]) > 0:
        truth_x = axis_X[truth_positions]
        truth_y = axis_Y[truth_positions]
        truth_z = 2 * axis_Z_max * np.ones_like(truth_x)  # 在最高点标记

        ax.scatter(truth_x, truth_y, truth_z, c='red', marker='o', s=100,
                   label='Ground truth', depthshade=False)

        # 绘制垂直线
        for i in range(len(truth_x)):
            ax.plot([truth_x[i], truth_x[i]], [truth_y[i], truth_y[i]], [0, truth_z[i]], color='black',
                    linewidth=0.5, linestyle='--')

    # 设置标签和标题
    ax.set_xlabel('Delay (norm)')
    ax.set_ylabel('Doppler shift (norm)')
    ax.set_zlabel('Peak Intensity')
    ax.set_title(custom_title)
    ax.legend()

    # 添加颜色条
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Intensity')

    # 设置视角
    ax.view_init(elev=30, azim=45)

    plt.tight_layout()
    plt.show()

    return fig, ax


def ground_truth_matrix(tau, f, axis_X, axis_Y, xstep, ystep):
    """
    在plotPeaks.m中调用，产生真实2维tau、f的stem图所用的矩阵数据
    横轴X为tau，纵轴Y为f
    """
    array_reg = np.zeros((axis_Y.shape[0], axis_X.shape[1]))

    for idx1 in range(axis_Y.shape[0]):
        for idx2 in range(axis_X.shape[1]):
            array_reg[idx1, idx2] = ground_truth_matrix_func(
                tau, f, axis_X[idx1, idx2], axis_Y[idx1, idx2], xstep, ystep
            )

    return array_reg


def ground_truth_matrix_func(tau, f, x, y, xstep, ystep):
    """
    peakSearchFunc
    横轴X为tau,纵轴Y为f。
    step为间隔，作为判定标准。
    """
    L = len(tau)
    x_std = xstep / 2
    y_std = ystep / 2

    reg = 0
    for idx in range(L):
        if abs(x - tau[idx]) < x_std and abs(y - f[idx]) < y_std:
            reg = 1

    return reg

def test_peak_searching():
    """测试峰值搜索功能"""
    print("测试峰值搜索算法...")

    # 生成测试数据
    np.random.seed(43)

    # 模拟ADMM算法输出的phi
    len_val = 400  # Nb * Nd = 20 * 20
    # phi = np.random.randn(len_val) + 1j * np.random.randn(len_val)
    phi = np.zeros(len_val)
    phi[2] = 1
    # phi = phi / np.linalg.norm(phi)  # 归一化

    # 设置函数参数
    func_opts = {
        'phi': phi,
        'xbase': 20,  # Nd
        'ybase': 20  # Nb
    }

    # 设置搜索参数
    search_opts = {
        'xmin': 0, 'xmax': 1, 'xstep': 0.02,
        'ymin': -0.5, 'ymax': 0.5, 'ystep': 0.02,
        'reducefactor': 0.1, 'iter': 2
    }

    # 执行峰值搜索
    peaks = alt_peak_search(func_opts, search_opts)
    print(f"找到 {len(peaks)} 个峰值:")
    for i, peak in enumerate(peaks):
        print(f"峰值 {i + 1}: τ={peak[0]:.3f}, f={peak[1]:.3f}, 强度={peak[2]:.3f}")

    return peaks, func_opts


def test_plot_peaks(peaks, func_opts):
    """测试峰值绘图功能"""
    print("\n测试峰值绘图...")

    # 生成真实峰值位置（这里用搜索到的峰值加上一些扰动来模拟）
    ground_truth = {
        'tau': np.array([0.25, 0.45, 0.75]),  # 示例真实时延
        'f': np.array([-0.2, 0.1, 0.3])  # 示例真实多普勒频移
    }

    # 绘图参数
    plot_opts = {
        'xmin': 0, 'xmax': 1, 'xstep': 0.01,
        'ymin': -0.5, 'ymax': 0.5, 'ystep': 0.01,
        'custom_title': 'Proposed Method - Peak Search Results',
        'custom_legend': 'Detected Peaks'
    }

    # 绘制峰值图
    fig, ax = plot_peaks(func_opts, ground_truth, plot_opts)

    return fig, ax




# 运行测试
if __name__ == "__main__":

    # 创建包含平台区域的图像
    image_with_plateau = np.array([
        [1, 1, 1, 2, 3],
        [1, 5, 5, 4, 3],  # 平台区域：两个相邻的5
        [2, 5, 5, 4, 2],
        [3, 4, 4, 3, 1]
    ])

    maxima = local_maxima(image_with_plateau, connectivity=2)
    print("平台区域处理结果：")
    print(maxima.astype(int))


    # 测试峰值搜索
    peaks, func_opts = test_peak_searching()

    # 测试绘图（需要3D绘图支持）
    try:
        fig, ax = test_plot_peaks(peaks, func_opts)
    except Exception as e:
        print(f"3D绘图失败: {e}")
        print("这可能是由于缺少3D支持，但2D功能应该正常")