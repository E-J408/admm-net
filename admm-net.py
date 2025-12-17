import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableMatrixInverse(nn.Module):
    """
    通过神经网络学习矩阵求逆的可微近似
    理论基础：Neumann级数展开
    """

    def __init__(self, dim, num_terms):
        super(DifferentiableMatrixInverse, self).__init__()
        self.dim = dim
        self.num_terms = num_terms
        # 学习Neumann级数的权重系数
        self.neumann_weights = nn.Parameter(torch.ones(num_terms))

    def forward(self, A):
        """
        近似计算（I+A）{-1}
        :param A: 正定矩阵 [batch_size, dim, dim]
        """
        batch_size = A.shape[0]
        I = torch.eye(self.dim, device=A.device).unsqueeze(0).repeat(batch_size, 1, 1)

        # Neumann级数近似
        result = I.clone()
        A_power = I.clone()

        for k in range(1, self.num_terms + 1):
            A_power = torch.bmm(A_power, A)
            weight = torch.sigmoid(self.neumann_weights[k - 1])  # 约束在[0,1]
            result = result + (-1) ** k * weight * A_power

        return result


class ComplexLinear(nn.Module):
    """
    复数线性变换的实值实现
    """

    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        # 实部权重 [out_features, in_features]
        self.weight_real = nn.Parameter(torch.Tensor(out_features, in_features))
        # 虚部权重
        self.weight_imag = nn.Parameter(torch.Tensor(out_features, in_features))
        # 偏置
        self.bias_real = nn.Parameter(torch.Tensor(out_features))
        self.bias_imag = nn.Parameter(torch.Tensor(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_real)
        nn.init.kaiming_uniform_(self.weight_imag)
        nn.init.zeros_(self.bias_real)
        nn.init.zeros_(self.bias_imag)

    def forward(self, x_real, x_imag):
        """复数矩阵乘法: (W_real + jW_imag)(x_real + jx_imag)"""
        out_real = F.linear(x_real, self.weight_real, self.bias_real) - \
                   F.linear(x_imag, self.weight_imag, torch.zeros_like(self.bias_real))
        out_imag = F.linear(x_real, self.weight_imag, self.bias_imag) + \
                   F.linear(x_imag, self.weight_real, torch.zeros_like(self.bias_imag))
        return out_real, out_imag


class PhiLayer(nn.Module):
    """Phi 对应层 解析解方案"""

    def __init__(self, epsilon=1e-8):
        super(PhiLayer, self).__init__()
        self.rho = nn.Parameter(torch.tensor(1.0))  # 可学习的ρ
        self.epsilon = epsilon  # 用于防止除零的小常数

    def forward(self, y, b, G, Z, k):
        """

        :param y: 观测向量，维度：[batch_size, MN]（复数）
        :param b: 解调符号，维度：[batch_size, MN]（复数）
        :param G: 辅助变量，维度：[batch_size, MN+1, MN+1]（复数）
        :param Z: 对偶变量，维度：[batch_size, MN+1, MN+1]（复数）
        :param k: 当前层索引
        """
        batch_size, dim = y.shape
        # 提取g^k和ζ^k
        g = G[:, :-1, -1]  # [batch_size, MN]
        zeta = Z[:, :-1, -1]  # [batch_size, MN]

        # 计算|b|²，加epsilon防止除零
        b_sq = torch.abs(b) ** 2 + self.epsilon  # [batch_size, MN]

        # 计算ρ (使用softplus函数来保证正数)
        rho_val = F.softplus(self.rho)  # rho > 0

        # 核心计算式
        weight = b_sq / (1 + rho_val * b_sq)
        y_over_b = y / (b + self.epsilon)
        right_term = y_over_b + rho_val * g + zeta
        phi_new = weight * right_term

        return phi_new


class HLayer(nn.Module):
    """
    H 对应层
    基于“对角矩阵松弛条件”的简化版本
    """

    def __init__(self, M, N, epsilon=1e-8):
        super(HLayer, self).__init__()
        self.M, self.N = M, N
        self.dim = M * N
        self.epsilon = epsilon

        # 可学习的惩罚项参数 rho
        self.rho = nn.Parameter(torch.tensor(1.0))

        # 可学习的投影参数，用于满足软约束
        self.projection_weight = nn.Parameter(torch.tensor(1.0))

        # 轻量网络，用于学习从“目标向量”到“可行向量”的修正
        self.correction_net = nn.Sequential(
            nn.Linear(self.dim, 64),  # 输入维度MN
            nn.ReLU(),
            nn.Linear(64, self.dim),  # 输出维度MN
            nn.Tanh()  # 输出范围[-1,1]
        )

    def forward(self, phi, G, Z, sigma, k):
        """

        :param phi:
        :param G: 辅助变量，维度：[batch_size, MN+1, MN+1]（复数）
        :param Z: 辅助变量，维度：[batch_size, MN+1, MN+1]（复数）
        :param sigma: 解调符号上限
        :param k: 层数
        """
        batch_size = G.shape[0]

        # --- 步骤一：提取目标向量 t = diag(G + Z/ρ)
        G_sub = G[:, :self.dim, :self.dim]  # [batch_size, MN, MN]
        Z_sub = Z[:, :self.dim, :self.dim]
        rho_val = F.softplus(self.rho)

        # 计算G+Z/ρ，并取其对角元素实部（因为H是实对角矩阵）
        T_matrix = G_sub + Z_sub / (rho_val + self.epsilon)
        t = torch.diagonal(T_matrix, dim1=1, dim2=2).real  # [batch_size, MN]

        # --- 步骤二：投影到可行集（软约束）
        # 可行集约束：(2√(MN)σ + σ²) * ||h||_∞ + sum(h) ≤ 1
        # 对于对角矩阵H=diag(h)，其谱范数||H||₂ = ||h||_∞，迹Tr(H) = sum(h)

        # 计算约束中的常数部分 A = (2√(MN)σ + σ²)
        A = 2 * torch.sqrt(torch.tensor(self.M * self.N).float()) * sigma + sigma ** 2
        A = A.view(-1, 1)  # 调整为[batch_size, 1]

        # 神经网络修正：让投影过程可学习
        correction = self.correction_net(t)  # 学习一个修正量
        t_corrected = t + 0.1 * correction  # 小幅修正，保持主方向

        # 核心投影操作：缩放t_corrected使其满足约束
        # 这是一个可微的近似投影，替代复杂的带约束优化
        h_projected = self._differentiable_projection(t_corrected, A, batch_size)

        # --- 步骤三： 构造对角矩阵 H = diag(h)
        H = torch.diag_embed(h_projected)  # 形状: [batch_size, MN, MN]

        return H

    def _differentiable_projection(self, v, A, batch_size):
        """
        可微的投影函数，将向量v投影到约束集合C
        C = { h | A * ||h||_∞ + sum(h) ≤ 1 }
        使用软投影而非硬裁剪，保证梯度流通。
        """
        # 计算当前向量的约束值
        l_inf_norm = torch.max(torch.abs(v), dim=1, keepdim=True)[0]  # L∞范数
        trace_v = torch.sum(v, dim=1, keepdim=True)  # 迹（和）
        constraint_val = A * l_inf_norm + trace_v  # 形状: [batch_size, 1]

        # 计算缩放因子：如果违反约束，则按比例缩小
        # 使用sigmoid确保缩放因子平滑且在(0,1)附近
        scale = torch.sigmoid(self.projection_weight) / (constraint_val + self.epsilon)
        scale = torch.clamp(scale, max=1.0)  # 不超过1，即只缩小不放大

        # 应用缩放（软投影）
        v_projected = v * scale

        return v_projected

    def get_constraint_info(self, h, sigma):
        """
        计算当前h向量的约束满足情况，用于监控训练
        """
        A = 2 * torch.sqrt(torch.tensor(self.M * self.N).float()) * sigma + sigma ** 2
        l_inf_norm = torch.max(torch.abs(h), dim=1)[0]
        trace_h = torch.sum(h, dim=1)
        constraint_val = A * l_inf_norm + trace_h
        violation = torch.relu(constraint_val - 1.0).mean()  # 违反程度
        return violation.item()


class GLayer(nn.Module):
    """G 对应层"""

    def __init__(self, M, N, epsilon=1e-8, use_learnable_threshold=True):
        super(GLayer, self).__init__()
        self.M, self.N = M, N
        self.dim = M * N + 1  # G是[MN+1, MN+1]的矩阵
        self.epsilon = epsilon

        # 可学习的正则化参数
        self.lambda_param = nn.Parameter(torch.tensor(0.1))

        # 可学习的惩罚参数rho
        self.rho = nn.Parameter(torch.tensor(1.0))

        # 可学习的特征值阈值，默认值为0
        if use_learnable_threshold:
            self.threshold = nn.Parameter(torch.tensor(0.0))
        else:
            self.threshold = torch.tensor(0.0)

        # 特征值修正网络，学习最优的非负化策略（存疑）
        self.value_net = nn.Sequential(
            nn.Linear(1, 16),  # 输入单个特征值
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # 输出[0,1]范围内的修正权重
        )

    def forward(self, phi, H, Z, k):
        """
        G层前向传播
        :param phi: 当前φ的估计值[batch_size, MN]
        :param H: 投影后的H矩阵[batch_size, MN, MN]
        :param Z: 对偶变量[batch_size, MN+1, MN+1]
        :param k: 当前层索引
        :return: 更新后的半正定矩阵[batch_size, MN+1, MN+1]
        """
        batch_size = phi.shape[0]

        # --- 步骤一：构造分块矩阵
        G_matrix = self._build_block_matrix(phi, H, Z)

        # --- 步骤二：特征分解
        values, vectors = self._eigen_decomposition(G_matrix, k)

        # --- 步骤三：特征修正
        values_corrected = self._eigenvalues_projection(values, k)

        # --- 步骤四：重构半正定矩阵
        G_updated = self._rebuild_definite_matrix(values_corrected, vectors)

        return G_updated

    def _build_block_matrix(self, phi, H, Z):
        """
        根据公式构造分块矩阵
        """
        batch_size = phi.shape[0]

        # 计算λ^{-2}（保证正性）
        lambda_val = F.softplus(self.lambda_param)  # lambda > 0
        lambda_inv = 1.0 / (lambda_val ** 2 + self.epsilon)  # λ^{-2}

        # 构造分块矩阵
        # 右上部分phi
        phi_ex = phi.unsqueeze(-1)  # [batch_size, MN, 1]
        # 左下部分phi^H
        phi_H = phi.conj().unsqueeze(1)  # [batch_size, 1, MN]
        # 右下部分
        lambda_matrix_full = torch.full((batch_size, 1, 1), lambda_inv,
                                        device=H.device, dtype=H.dtype)
        # 组合
        top = torch.cat([H, phi_ex], dim=2)  # [batch_size, MN, MN+1]
        bottom = torch.cat([phi_H, lambda_matrix_full], dim=2)  # [batch_size, 1, MN+1]
        block_matrix = torch.cat([top, bottom], dim=1)  # [batch_size, MN+1, MN+1]

        # 相减运算
        rho_val = F.softplus(self.rho)
        G_matrix = block_matrix - (1.0 / (rho_val + self.epsilon)) * Z

        return G_matrix

    def _eigen_decomposition(self, matrix, k):
        """
        特征分解
        :param matrix: 构造的分块矩阵
        :param k: 网络层数
        :return: values, vectors: 特征值和特征向量
        """

        # 确保矩阵是Hermitian的（强制对称）
        matrix_hermitian = 0.5 * (matrix + matrix.transpose(1, 2).conj())

        values, vectors = torch.linalg.eigh(matrix_hermitian)

        return values, vectors

    def _eigenvalues_projection(self, values, k):
        """
        可学习的特征值处理
        替代固定的max(0, λ)操作
        :param values: 特征值矩阵 [batch_size, dim]
        :param k: 层数
        :return: 处理过的特征值矩阵 [batch_size, dim]
        """
        batch_size, dim = values.shape

        # 可学习阈值
        threshold = torch.sigmoid(self.threshold)

        # 对每个特征值独立处理
        processed_values = []
        for i in range(dim):
            eig_val = values[:, i:i + 1]  # 当前特征值 [batch_size, 1]
            # 非负化
            base = F.softplus(eig_val - threshold)
            # 可学习的缩放因子
            scale = self.value_net(eig_val.abs())
            eig_processed = base * scale
            processed_values.append(eig_processed)

        return torch.cat(processed_values, dim=1)  # [batch_size, dim]

    def _rebuild_definite_matrix(self, values, vectors):
        """
        重构半正定矩阵
        :param values: 特征值矩阵 [batch_size, dim]
        :param vectors: 特征向量矩阵 [batch_size, dim, dim]
        :return: 该轮次的G [batch_size, dim, dim]
        """
        # 构造对角特征值矩阵
        eig_diag = torch.diag_embed(values)

        # 计算 G = U Λ U^H
        G = torch.bmm(vectors, torch.bmm(eig_diag, vectors.tanspose(1, 2).conj()))

        # 强制Hermitian
        G = 0.5 * (G + G.transpose(1, 2).conj())

        return G


class ZLayer(nn.Module):
    """Z 对应层"""

    def __init__(self, M, N):
        super(ZLayer, self).__init__()

    def forward(self, x):
        """前向传播"""
        pass


class PeakSearchLayer(nn.Module):
    """峰值搜索层"""

    def __init__(self, M, N, L):
        super(PeakSearchLayer, self).__init__()

    def forward(self, x):
        """前向传播"""
        pass


class ADMMNet(nn.Module):
    """完整的ADMM-Net模型"""

    def __init__(self, M, N, L=3, num_layers=10):
        super(ADMMNet, self).__init__()
        self.num_layers = num_layers
        self.M, self.N, self.L = M, N, L

        # 初始化网络组件
        self.phiLayers = nn.ModuleList([
            PhiLayer() for _ in range(num_layers)
        ])
        self.hLayers = nn.ModuleList([
            HLayer(M, N) for _ in range(num_layers)
        ])
        self.gLayers = nn.ModuleList([
            GLayer(M, N) for _ in range(num_layers)
        ])
        self.zLayers = nn.ModuleList([
            ZLayer(M, N) for _ in range(num_layers)
        ])

        self.peakSearchLayer = PeakSearchLayer(M, N, L)

    def forward(self, y, b, sigma):
        """

        :param y: 观测向量
        :param b: 解调符号
        :param sigma: 噪声上限
        """
        batch_size = y.shape[0]
        M = self.M
        N = self.N
        G = torch.zeros(batch_size, M * N + 1, M * N + 1)
        Z = torch.zeros(batch_size, M * N + 1, M * N + 1)
        phi = torch.zeros(batch_size, M * N)

        for k in range(self.num_layers):
            # 第l层前向传播
            phi = self.phiLayers[k](y, b, G, Z, k)
            H = self.hLayers[k](phi, G, Z, sigma, k)
            G = self.gLayers[k](phi, H, Z, k)
            Z = self.zLayers[k](phi, H, G, Z, k)

        # 谱峰搜索
        tau_est, f_est = self.peakSearchLayer(phi, b)

        return tau_est, f_est
