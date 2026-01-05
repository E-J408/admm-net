import numpy as np
import cvxpy as cp
from scipy.linalg import svd


def admm_for_us(y, b, xbase, ybase, lambda_val, sigma, opts=None, use_min_iter=True, min_iter=5):
    """
    Solve the joint delay-Doppler atomic minimization problem using ADMM
    使用ADMM解决时延-多普勒联合原子范数最小化问题

    Parameters:
    -----------
    y : ndarray, shape (len, )
        Demodulated signal result
    b : ndarray, shape (len, )
        Demodulated symbols
    xbase, ybase : int
        Structure values, xbase * ybase = len
    lambda_val, sigma : float
        Hyperparameters
    opts : dict, optional
        Control variables:
        - rho: hyperparameter > 0
        - max_iter: maximum iterations
        - eta_abs, eta_rel: stopping criteria parameters

    Returns:
    --------
    phi : ndarray, shape (len, )
        Optimization result for peak searching
    iter_count : int
        Number of iterations
    """

    # Default parameters
    rho = 1.0
    max_iter = 500
    eta_abs = 1e-5
    eta_rel = 1e-5

    if opts is not None:
        rho = opts.get('rho', rho)
        max_iter = opts.get('max_iter', max_iter)
        eta_abs = opts.get('eta_abs', eta_abs)
        eta_rel = opts.get('eta_rel', eta_rel)

    # 修复，确保y和b为向量输入
    y = y.flatten()
    b = b.flatten()

    len_val = y.shape[0]  # len即MN，为y和b的长度

    # Initialize variables
    G0 = np.zeros((len_val + 1, len_val + 1), dtype=complex)
    Z0 = np.zeros((len_val + 1, len_val + 1), dtype=complex)

    HK = np.zeros((len_val, len_val), dtype=complex)

    phiK = np.zeros(len_val, dtype=complex)
    # 调试信息
    print(f"Starting ADMM with len_val={len_val}, max_iter={max_iter}, eta_abs={eta_abs}")
    iter_count = 0
    for iter_count in range(1, max_iter + 1):
        if iter_count == 1:
            GK = G0.copy()
            ZK = Z0.copy()
            HK_pre = np.zeros((len_val, len_val), dtype=complex)
        else:
            HK_pre = HK.copy()  # 保存上一份HK，用于计算stopping criteria

        GK_hat = GK[:len_val, :len_val]
        gK = GK[:len_val, len_val]
        ZK_hat = ZK[:len_val, :len_val]
        zetaK = ZK[:len_val, len_val]

        # update phi
        b_conj = np.conj(b)
        diag_inv = np.linalg.inv(np.diag(b * b_conj)) + rho * np.ones(len_val)
        phiK = np.linalg.inv(diag_inv) @ (np.linalg.inv(np.diag(b)) @ y + rho * gK + zetaK)

        # update H (采用简化的H版本)
        HK = admm_for_us_H_cvx_0(GK_hat, ZK_hat, rho, xbase, ybase, sigma)

        # update G (采用SVD方法)
        GK = admm_for_us_G_svd(HK, phiK, lambda_val, ZK, rho)

        # update Z
        HK_phi_block = np.vstack([
            np.hstack([HK, phiK.reshape(-1, 1)]),
            np.hstack([phiK.conj().T, 1.0 / (lambda_val ** 2)])
        ])
        ZK = ZK + rho * (GK - HK_phi_block)

        # 设置最小迭代次数
        if use_min_iter and iter_count < min_iter:
            continue

        # stopping criteria
        if iter_count > 1:
            norm_GK = np.linalg.norm(GK, 'fro')
            norm_HK_phi = np.linalg.norm(HK_phi_block, 'fro')
            eta_pri_val = eta_abs * np.sqrt(len_val + 1) + eta_rel * max(norm_GK, norm_HK_phi)

            norm_ZK = np.linalg.norm(ZK, 'fro')
            eta_dual_val = eta_abs * np.sqrt(len_val) + eta_rel * norm_ZK

            residual_pri = np.linalg.norm(GK - HK_phi_block, 'fro')
            residual_dual = np.linalg.norm(rho * (HK - HK_pre), 'fro')

            if residual_pri <= eta_pri_val and residual_dual <= eta_dual_val:
                print(f"退出admm迭代，当前迭代次数为: {iter_count}")
                break
    phi = phiK
    return phi, iter_count


def admm_for_us_H_cvx_0(GK_hat, ZK_hat, rho, xbase, ybase, sigma):
    """
    简化H版本 - H是一个对角矩阵，对角元均为实数
    """
    Nd = xbase
    Nb = ybase
    len_val = Nb * Nd

    # 提取对角线元素
    diag_GZ = np.diag(GK_hat + ZK_hat / rho)

    # 使用cvxpy进行优化
    H = cp.Variable(len_val)

    # 目标函数：最小化 ||H - diag(GK_hat + ZK_hat/rho)||_F
    objective = cp.Minimize(cp.norm(H - diag_GZ, 'fro'))

    # 约束条件
    constraints = [
        cp.norm(H, 'inf') * (2 * np.sqrt(Nb*Nd) * sigma + sigma**2) + cp.sum(H) <= 1
    ]

    # 求解
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS, verbose=False)

    # 检查求解状态
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print(f"警告: 求解器返回状态: {prob.status}")

    H_opt = H.value
    return np.diag(H_opt)


def admm_for_us_G_svd(HK, phiK, lambda_val, ZK, rho):
    """
    SVD方法计算G
    """
    len_val = HK.shape[0]

    # 构造sd_Matrix
    sd_Matrix = np.zeros((len_val + 1, len_val + 1), dtype=complex)
    sd_Matrix[:len_val, :len_val] = HK
    sd_Matrix[:len_val, len_val] = phiK
    sd_Matrix[len_val, :len_val] = phiK.conj().T
    sd_Matrix[len_val, len_val] = 1.0 / (lambda_val ** 2)

    sd_Matrix = sd_Matrix - ZK / rho

    # SVD分解，显式指定
    U, S_diag, Vh = svd(sd_Matrix)

    # 将小于0的奇异值置0
    S_diag[S_diag < 0] = 0

    # 重建奇异值矩阵
    S = np.zeros_like(sd_Matrix, dtype=complex)
    np.fill_diagonal(S, S_diag)

    # 重构矩阵
    GK = U @ S @ Vh

    return GK


def admm_for_us_G_cvx(HK, phiK, lambda_val, ZK, rho):
    """
    CVX计算版本（备用方法）
    """
    len_val = HK.shape[0]

    # 构造sd_Matrix
    sd_Matrix = np.zeros((len_val + 1, len_val + 1), dtype=complex)
    sd_Matrix[:len_val, :len_val] = HK
    sd_Matrix[:len_val, len_val] = phiK
    sd_Matrix[len_val, :len_val] = phiK.conj().T
    sd_Matrix[len_val, len_val] = 1.0 / (lambda_val ** 2)

    sd_Matrix = sd_Matrix - ZK / rho

    # 使用cvxpy进行优化
    G = cp.Variable((len_val + 1, len_val + 1), hermitian=True)

    # 目标函数
    objective = cp.Minimize(cp.norm(G - sd_Matrix, 'fro'))

    # 约束条件：G >= 0 (半正定)
    constraints = [G >> 0]

    # 求解
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        print(f"Warning: G optimization problem status: {prob.status}")
        # 使用SVD方法作为备选
        return admm_for_us_G_svd(HK, phiK, lambda_val, ZK, rho)

    return G.value


# 测试函数
def test_admm():
    """测试ADMM函数"""
    # 生成测试数据
    np.random.seed(42)
    len_val = 20
    y = np.random.randn(len_val) + 1j * np.random.randn(len_val)
    b = np.random.randn(len_val) + 1j * np.random.randn(len_val)
    xbase, ybase = 4, 5  # 4 * 5=20

    lambda_val = 1.0
    sigma = 2.0

    opts = {
        'rho': 1.0,
        'max_iter': 100,
        'eta_abs': 1e-5,
        'eta_rel': 1e-5
    }

    print("开始测试ADMM算法...")
    phi, iterations = admm_for_us(y, b, xbase, ybase, lambda_val, sigma, opts)

    print(f"算法完成，迭代次数: {iterations}")
    print(f"结果phi的形状: {phi.shape}")
    print(f"phi的前5个元素: {phi[:5]}")

    return phi, iterations


if __name__ == "__main__":
    # 运行测试
    test_admm()
