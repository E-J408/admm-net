import numpy as np


def vander_vec(start, step, length):
    """
    替代MATLAB中的vander_vec函数，生成范德蒙向量

    参数:
        start: 起始值（在当前实现中未使用）
        step: 步长值，用于计算指数部分的系数
        length: 向量长度，决定生成数组的元素个数

    返回值:
        numpy.ndarray: 复数类型的范德蒙向量，包含length个元素
    """
    # 生成范德蒙向量：e^(j*2*π*step*[0,1,2,...,length-1])
    return np.exp(1j * 2 * np.pi * step * np.arange(length))


def kr(A, B):
    """Khatri-Rao积（列wise Kronecker积）

    计算两个矩阵的Khatri-Rao积，即对应列的Kronecker积组成的矩阵

    参数:
        A (numpy.ndarray): 第一个输入矩阵，形状为(m, n)
        B (numpy.ndarray): 第二个输入矩阵，形状为(p, n)

    返回:
        numpy.ndarray: Khatri-Rao积结果矩阵，形状为(m*p, n)，数据类型为复数

    异常:
        ValueError: 当两个矩阵的列数不匹配时抛出
    """
    """Khatri-Rao积（列wise Kronecker积）"""
    m, n = A.shape
    p, n2 = B.shape
    if n != n2:
        raise ValueError("矩阵列数不匹配")

    # 初始化结果矩阵C，大小为(m*p, n)
    C = np.zeros((m * p, n), dtype=complex)
    # 对每一列计算Kronecker积并存入结果矩阵
    for i in range(n):
        C[:, i] = np.kron(A[:, i].reshape(-1, 1), B[:, i].reshape(-1, 1)).flatten()
    return C


def pskmod(data, M, phase_offset=0):
    """
    PSK调制函数

    对输入数据进行M进制相移键控调制，将数字信号转换为复数形式的模拟信号

    参数:
        data: 输入的数字数据，通常为0到M-1之间的整数数组
        M: 调制阶数，表示PSK调制的类型（如M=2为BPSK，M=4为QPSK等）
        phase_offset: 相位偏移量，用于调整星座图的初始相位，默认为0

    返回值:
        复数数组，表示调制后的信号，每个数据点对应星座图上的一个点
    """
    # 根据PSK调制公式计算复数信号：exp(j*(2π*data/M + phase_offset))
    return np.exp(1j * (2 * np.pi * data / M + phase_offset))


def pskdemod(sig, M, phase_offset=0):
    """
    PSK解调函数

    该函数对接收到的PSK调制信号进行解调，将其转换为对应的数字符号。

    参数:
        sig: 复数信号数组，表示接收到的PSK调制信号
        M: 调制阶数，如BPSK(M=2)、QPSK(M=4)等
        phase_offset: 相位偏移量，默认为0，用于补偿系统相位偏差

    返回值:
        解调后的数字符号数组，每个符号的取值范围为[0, M-1]
    """
    # 计算信号相位并减去相位偏移
    angles = np.angle(sig) - phase_offset
    # 将相位角度映射到[0, 2π)范围内
    angles = np.mod(angles + np.pi / M, 2 * np.pi)  # 调整相位边界
    # 将相位角度量化为对应的符号值
    return np.floor(angles * M / (2 * np.pi)).astype(int) % M


def awgn(sig, snr):
    """
    向信号添加加性高斯白噪声(AWGN)

    参数:
        sig: 输入信号，可以是实数或复数数组
        snr: 信噪比，单位为dB

    返回值:
        添加噪声后的信号数组
    """
    # 计算信号功率
    sig_power = np.mean(np.abs(sig) ** 2)
    # 根据信噪比计算噪声功率
    noise_power = sig_power / (10 ** (snr / 10))
    # 生成复高斯噪声
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(sig)) + 1j * np.random.randn(len(sig)))
    # 将噪声添加到原始信号上
    return sig + noise
