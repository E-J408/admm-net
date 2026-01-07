from pathlib import Path
from utils.mathUtils import *
from utils.peakSearchUtils import *
from admm import *
from utils.plotUtils import plot_predictions_vs_truth


def main():
    Path("./data").mkdir(exist_ok=True)
    # 基本参数
    Nb = 10  # OFDM block的数目
    Nd = 10  # 单个OFDM block内的数据符号数目

    L = 3  # 目标数目
    f = np.array([-0.25, 0, 0.14])  # 目标归一化多普勒频移
    tau = np.array([0.45, 0.25, 0.63])  # 目标归一化时延
    C = np.array([-0.5 + 1j, 0.6 - 0.2j, 0.3 + 0.7j])  # 目标反射路径加权

    # 生成S和D矩阵
    S = np.zeros((Nb, L), dtype=complex)
    D = np.zeros((Nd, L), dtype=complex)

    for i in range(L):
        S[:, i] = vander_vec(0, (Nb - 1) * f[i], Nb).reshape(-1)
        D[:, i] = vander_vec(0, (Nd - 1) * tau[i], Nd).reshape(-1)

    print(f"c shape:{C.shape}")
    print(f"C reshape:{C.reshape(-1, 1).shape}")
    Psi = kr(S, np.conj(D)) @ C.reshape(-1, 1)

    snr_e = 7  # 解调信噪比
    snr_w = 20  # 环境信噪比
    M_psk = 4  # QPSK调制

    data_type = 0

    if data_type == 0:
        # 重置原始数据
        data = np.random.randint(0, M_psk, Nb * Nd)
        sig = pskmod(data, M_psk, np.pi / M_psk)  # 真实调制符号 sig=b+e
        sig_n = awgn(sig, snr_e)  # 被解调噪声污染了的调制符号
        data_d = pskdemod(sig_n, M_psk, np.pi / M_psk)
        b = pskmod(data_d, M_psk, np.pi / M_psk)
        e = sig - b
        ser = 100 * np.sum(np.abs(e) > 1e-10) / len(e)

        # 保存数据
        np.savez('./data/data.npz', sig=sig, e=e)
        print(f"SER: {ser:.2f}%")

    elif data_type == 1:
        # 加载固定数据
        data_file = np.load('./data/data.npz')
        sig = data_file['sig']
        e = data_file['e']
        sig_n = awgn(sig, snr_e)
        data_d = pskdemod(sig_n, M_psk, np.pi / M_psk)
        b = pskmod(data_d, M_psk, np.pi / M_psk)
        e = sig - b  # 重新计算e
        ser = 100 * np.sum(np.abs(e) > 1e-10) / len(e)
        print(f"SER: {ser:.2f}%")

    elif data_type == 2:
        # 使用固定的e
        data_file = np.load('./data/data.npz')
        sig = data_file['sig']
        e = data_file['e']
        b = sig - e
        ser = 100 * np.sum(np.abs(e) > 1e-10) / len(e)
        print(f"SER: {ser:.2f}%")
    else:
        return

    # 生成所需数据
    real_y = np.diag(b + e) @ Psi
    w = np.sqrt(1 / 2) * (np.random.randn(Nb * Nd, 1) + 1j * np.random.randn(Nb * Nd, 1))
    w_var = np.linalg.norm(real_y) ** 2 / (10 ** (snr_w / 10) * Nb * Nd)
    y = real_y + np.sqrt(w_var) * w  # 对应论文中的式3.10

    # ANM-DUMV参数
    lambda_val = 1
    sigma = np.linalg.norm(e / b) + 1

    print("数据生成完成！")
    print(f"y的形状: {y.shape}")
    print(f"sigma值: {sigma:.4f}")

    opts = {
        'eta_abs': 1e-7,
        'eta_rel': 1e-7,
        'max_iter': 100
    }
    print(f"y的维度: {y.flatten().shape}")
    print(f"b的维度: {b.shape}")
    phi, iterations = admm_for_us(y, b, Nd, Nb, lambda_val, sigma, opts)
    print(phi.shape)

    print(f"算法完成，迭代次数: {iterations}")


    # 寻峰
    alt_peak_search_base = {
        'phi': phi,
        'xbase': Nb,
        'ybase': Nd,
    }
    alt_peak_search_opts = {
        'xstep': 1/(10*Nd),
        'ystep': 1/(10*Nb),
        'iter': 3
    }
    res_admm = alt_peak_search(alt_peak_search_base, alt_peak_search_opts)
    # 按照峰值大小排序
    res_admm = sorted(res_admm, key=lambda x: x[2], reverse=True)

    print(f"找到 {len(res_admm)} 个峰值:")
    for i, peak in enumerate(res_admm):
        print(f"{i + 1}. {peak}")
    # 按照数量过滤
    res_admm = res_admm[:3]
    plot_predictions_vs_truth(f, tau, res_admm)
    ground_truth_dict = {
        'f': f,
        'tau': tau,
    }
    plot_peaks(alt_peak_search_base, ground_truth_dict)








# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    main()
