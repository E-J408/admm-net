from utils.peakSearchUtils import *
import scipy.io as sio


def main():
    mat_data = sio.loadmat('data/mat/phi_ad.mat')
    phi = mat_data['phi_ad']
    print(phi.shape)
    # 转(100,1)为(100,)
    phi = phi.reshape(-1)
    print(phi.shape)

    Nb = 10
    Nd = 10

    # 寻峰
    alt_peak_search_base = {
        'phi': phi,
        'xbase': Nb,
        'ybase': Nd,
    }
    alt_peak_search_opts = {
        'xstep': 0.01,
        'ystep': 0.01,
        'iter': 3
    }
    res_admm = alt_peak_search(alt_peak_search_base, alt_peak_search_opts)
    # 按照峰值大小排序
    res_admm = sorted(res_admm, key=lambda x: x[2], reverse=True)

    print(f"找到 {len(res_admm)} 个峰值:")
    for i, peak in enumerate(res_admm):
        print(f"{i + 1}. {peak}")







# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    main()
