from scipy.linalg import khatri_rao
from pathlib import Path
import json
from typing import Dict, Tuple
from utils.mathUtils import *
from admm import *


class OFDMDatasetGenerator:
    """
    OFDM雷达数据集生成器 - 专为ANM-DUMV-ADMM网络设计
    生成标准化的训练、验证、测试数据集
    """

    def __init__(self, Nb=10, Nd=10, L_max=3, snr_range=(5, 25),
                 data_dir='./ofdm_dataset'):
        """
        参数初始化
        """
        self.Nb = Nb  # OFDM块数量
        self.Nd = Nd  # 每个块的数据符号数
        self.L_max = L_max  # 最大目标数
        self.snr_range = snr_range  # 信噪比范围(dB)
        self.data_dir = Path(data_dir)

        # 创建数据目录
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # 物理参数约束
        self.tau_range = (0.1, 0.9)  # 时延合理范围
        self.f_range = (-0.4, 0.4)  # 多普勒合理范围

        # 数据集配置
        self.dataset_config = {
            'Nb': Nb,
            'Nd': Nd,
            'L_max': L_max,
            'snr_range': snr_range,
            'total_samples': 0,
            'train_samples': 0,
            'val_samples': 0,
            'test_samples': 0,
            'created_date': None
        }

    def generate_complete_dataset(self, total_samples=10000,
                                  train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        生成完整的数据集（训练集 + 验证集 + 测试集）
        """
        print("开始生成OFDM雷达数据集...")

        # 计算各集合样本数
        n_train = int(total_samples * train_ratio)
        n_val = int(total_samples * val_ratio)
        n_test = total_samples - n_train - n_val

        # 更新配置
        self.dataset_config.update({
            'total_samples': total_samples,
            'train_samples': n_train,
            'val_samples': n_val,
            'test_samples': n_test,
            'created_date': str(np.datetime64('now'))
        })

        # 生成各数据集
        print(f"生成训练集 ({n_train} 样本)...")
        train_data = self._generate_dataset_split(n_train, 'train')

        print(f"生成验证集 ({n_val} 样本)...")
        val_data = self._generate_dataset_split(n_val, 'val')

        print(f"生成测试集 ({n_test} 样本)...")
        test_data = self._generate_dataset_split(n_test, 'test')

        # 保存数据集
        self._save_dataset(train_data, val_data, test_data)

        # 保存配置文件
        self._save_dataset_config()

        print("数据集生成完成！")
        return train_data, val_data, test_data

    def _generate_dataset_split(self, n_samples: int, split_name: str) -> Dict[str, np.ndarray]:
        """
        生成指定数量的数据集分割
        """
        # 初始化存储数组
        data_dict = {
            'y_real': np.zeros((n_samples, self.Nb * self.Nd), dtype=np.float32),
            'y_imag': np.zeros((n_samples, self.Nb * self.Nd), dtype=np.float32),
            'b_real': np.zeros((n_samples, self.Nb * self.Nd), dtype=np.float32),
            'b_imag': np.zeros((n_samples, self.Nb * self.Nd), dtype=np.float32),
            'tau': np.zeros((n_samples, self.L_max), dtype=np.float32),
            'f': np.zeros((n_samples, self.L_max), dtype=np.float32),
            'C_real': np.zeros((n_samples, self.L_max), dtype=np.float32),
            'C_imag': np.zeros((n_samples, self.L_max), dtype=np.float32),
            'L_true': np.zeros((n_samples,), dtype=np.int32),
            'sigma': np.zeros((n_samples,), dtype=np.float32),
            'ser': np.zeros((n_samples,), dtype=np.float32)
        }

        for i in range(n_samples):
            if i % 1000 == 0:
                print(f"  {split_name}: 已生成 {i}/{n_samples} 样本")

            # # 随机生成目标数量 (1到L_max)
            # L = np.random.randint(1, self.L_max + 1)

            # 生成目标数量
            L = self.L_max

            # 生成单个样本
            sample = self._generate_single_sample(L)

            # 存储到数组
            data_dict['y_real'][i] = sample['y'].real
            data_dict['y_imag'][i] = sample['y'].imag
            data_dict['b_real'][i] = sample['b'].real
            data_dict['b_imag'][i] = sample['b'].imag
            data_dict['tau'][i, :L] = sample['tau']
            data_dict['f'][i, :L] = sample['f']
            data_dict['C_real'][i, :L] = sample['C'].real
            data_dict['C_imag'][i, :L] = sample['C'].imag
            data_dict['L_true'][i] = L
            data_dict['sigma'][i] = sample['sigma']
            data_dict['ser'][i] = sample['ser']

        return data_dict

    def _generate_single_sample(self, L: int) -> Dict:
        """
        生成单个数据样本（基于您提供的代码结构）
        """
        # 1. 生成随机目标参数
        tau = np.random.uniform(*self.tau_range, L)
        f = np.random.uniform(*self.f_range, L)

        # 2. 生成复反射系数
        C_real = np.random.normal(0, 0.7, L)
        C_imag = np.random.normal(0, 0.7, L)
        C = C_real + 1j * C_imag

        # 3. 生成S和D矩阵（VanderMonde结构）
        S = np.zeros((self.Nb, L), dtype=complex)
        D = np.zeros((self.Nd, L), dtype=complex)

        for i in range(L):
            S[:, i] = vander_vec(0, (self.Nb - 1) * f[i], self.Nb).reshape(-1)
            D[:, i] = vander_vec(0, (self.Nd - 1) * tau[i], self.Nd).reshape(-1)

        # 4. 构建Psi矩阵（Khatri-Rao积）
        Psi = kr(S, np.conj(D)) @ C.reshape(-1, 1)

        # 5. 生成通信符号（QPSK调制）
        b, e, ser = self._generate_communication_symbols()

        # 6. 生成观测信号y
        real_y = np.diag(b + e) @ Psi

        # 7. 添加噪声
        snr_w = np.random.uniform(*self.snr_range)
        w = np.sqrt(1 / 2) * (np.random.randn(self.Nb * self.Nd, 1) + 1j * np.random.randn(self.Nb * self.Nd, 1))
        w_var = np.linalg.norm(real_y) ** 2 / (10 ** (snr_w / 10) * self.Nb * self.Nd)

        y = real_y + np.sqrt(w_var) * w
        y = y.flatten()  # 转为向量输入

        sigma = np.linalg.norm(e / b) + 1

        # # 3. 生成S和D矩阵（VanderMonde结构）
        # n_range_s = np.arange(self.Nb).reshape(-1, 1)  # 形状为(Nb, 1)
        # n_range_d = np.arange(self.Nd).reshape(-1, 1)  # 形状为(Nd, 1)
        # S = np.exp(1j * 2 * np.pi * n_range_s * f)  # 形状为(Nb, L)
        # D = np.exp(1j * 2 * np.pi * n_range_d * tau)  # 形状为(Nd, L)
        # # 4. 构建Psi矩阵（Khatri-Rao积）
        # Psi = khatri_rao(S, D.conj()) @ C.reshape(-1, 1)
        #
        # # 5. 生成通信符号（QPSK调制）
        # b, e, ser = self._generate_communication_symbols()
        #
        # # 6. 生成观测信号y
        # real_y = np.diag(b + e) @ Psi.squeeze()
        #
        # # 7. 添加噪声
        # snr_db = np.random.uniform(*self.snr_range)
        # sigma = 10 ** (-snr_db / 20)
        # noise = np.random.randn(*real_y.shape) * sigma / np.sqrt(2)
        # noise = noise.astype(np.complex64)
        #
        # y = real_y + noise

        return {
            'y': y.astype(np.complex64),
            'b': b.astype(np.complex64),
            'tau': tau.astype(np.float32),
            'f': f.astype(np.float32),
            'C': C.astype(np.complex64),
            'sigma': np.float32(sigma),
            'ser': ser
        }

    def _generate_communication_symbols(self, M_psk: int = 4) -> Tuple[np.ndarray, np.ndarray, float]:
        """生成通信符号（考虑解调误差）"""
        # 生成随机数据
        data = np.random.randint(0, M_psk, self.Nb * self.Nd)

        sig = pskmod(data, M_psk, np.pi / M_psk)

        # 添加解调误差
        snr_e = 7  # 解调信噪比
        sig_n = awgn(sig, snr_e)
        data_d = pskdemod(sig_n, M_psk, np.pi / M_psk)
        b = pskmod(data_d, M_psk, np.pi / M_psk)
        # 计算解调误差
        e = sig - b
        ser = 100 * np.sum(np.abs(e) > 1e-10) / len(e) # SER%

        return b, e, ser

    def _save_dataset(self, train_data: Dict, val_data: Dict, test_data: Dict):
        """保存数据集到本地文件[6,7](@ref)"""

        # 保存训练集
        train_dir = self.data_dir / 'train'
        train_dir.mkdir(exist_ok=True)

        for key, array in train_data.items():
            np.save(train_dir / f'{key}.npy', array)

        # 保存验证集
        val_dir = self.data_dir / 'val'
        val_dir.mkdir(exist_ok=True)

        for key, array in val_data.items():
            np.save(val_dir / f'{key}.npy', array)

        # 保存测试集
        test_dir = self.data_dir / 'test'
        test_dir.mkdir(exist_ok=True)

        for key, array in test_data.items():
            np.save(test_dir / f'{key}.npy', array)

        print(f"数据集已保存到: {self.data_dir}")

    def _save_dataset_config(self):
        """保存数据集配置文件"""
        config_path = self.data_dir / 'dataset_config.json'
        with open(config_path, 'w') as f:
            json.dump(self.dataset_config, f, indent=2)

        # 同时保存为Python可读的NPZ文件
        np.savez(self.data_dir / 'dataset_info.npz', **self.dataset_config)

    def create_pytorch_dataloader(self, batch_size=32, split='train', shuffle=True):
        """
        创建PyTorch DataLoader以便直接用于训练
        """
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        # 加载数据
        split_dir = self.data_dir / split
        if not split_dir.exists():
            raise ValueError(f"分割 {split} 不存在，请先生成数据集")

        # 加载所有npy文件
        data_arrays = {}
        for file in split_dir.glob('*.npy'):
            key = file.stem
            data_arrays[key] = np.load(file)

        # 将复数数据组合成复数张量
        y_real = torch.FloatTensor(data_arrays['y_real'])
        y_imag = torch.FloatTensor(data_arrays['y_imag'])
        y = torch.complex(y_real, y_imag)

        b_real = torch.FloatTensor(data_arrays['b_real'])
        b_imag = torch.FloatTensor(data_arrays['b_imag'])
        b = torch.complex(b_real, b_imag)

        # 其他数据
        tau = torch.FloatTensor(data_arrays['tau'])
        f = torch.FloatTensor(data_arrays['f'])
        C_real = torch.FloatTensor(data_arrays['C_real'])
        C_imag = torch.FloatTensor(data_arrays['C_imag'])
        C = torch.complex(C_real, C_imag)
        L_true = torch.LongTensor(data_arrays['L_true'])
        sigma = torch.FloatTensor(data_arrays['sigma'])

        # 创建TensorDataset
        dataset = TensorDataset(y, b, tau, f, C, L_true, sigma)

        # 创建DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return dataloader

    def visualize_dataset_stats(self):
        """可视化数据集统计信息"""
        import matplotlib.pyplot as plt

        # 设置中文
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        # 加载配置
        config_path = self.data_dir / 'dataset_config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)

        # 加载训练集的目标数量分布
        L_true = np.load(self.data_dir / 'train' / 'L_true.npy')

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 目标数量分布
        unique, counts = np.unique(L_true, return_counts=True)
        axes[0, 0].bar(unique, counts)
        axes[0, 0].set_title('目标数量分布')
        axes[0, 0].set_xlabel('目标数量')
        axes[0, 0].set_ylabel('频次')

        # 时延分布
        tau = np.load(self.data_dir / 'train' / 'tau.npy')
        axes[0, 1].hist(tau.flatten(), bins=50, alpha=0.7)
        axes[0, 1].set_title('时延分布')
        axes[0, 1].set_xlabel('时延 τ')
        axes[0, 1].set_ylabel('频次')

        # 多普勒频率分布
        f = np.load(self.data_dir / 'train' / 'f.npy')
        axes[1, 0].hist(f.flatten(), bins=50, alpha=0.7)
        axes[1, 0].set_title('多普勒频率分布')
        axes[1, 0].set_xlabel('多普勒频率 f')
        axes[1, 0].set_ylabel('频次')

        # SER分布
        ser = np.load(self.data_dir / 'train' / 'ser.npy')
        axes[1, 1].hist(ser, bins=20, alpha=0.7)
        axes[1, 1].set_title('误符号率分布')
        axes[1, 1].set_xlabel('%')
        axes[1, 1].set_ylabel('频次')

        plt.tight_layout()
        plt.savefig(self.data_dir / 'dataset_statistics.png', dpi=300, bbox_inches='tight')
        plt.show()


class DatasetGeneratorCreatePhi(OFDMDatasetGenerator):
    """
    生成用传统方法计算phi的数据集
    """

    def __init__(self, Nb=10, Nd=10, L_max=3, snr_range=(5, 25),
                 data_dir='./ofdm_dataset'):
        super().__init__(Nb, Nd, L_max, snr_range, data_dir)

    def _generate_dataset_split(self, n_samples: int, split_name: str) -> Dict[str, np.ndarray]:
        """生成一个数据集分片"""
        # 初始化存储数组
        data_dict = {
            'y_real': np.zeros((n_samples, self.Nb * self.Nd), dtype=np.float32),
            'y_imag': np.zeros((n_samples, self.Nb * self.Nd), dtype=np.float32),
            'b_real': np.zeros((n_samples, self.Nb * self.Nd), dtype=np.float32),
            'b_imag': np.zeros((n_samples, self.Nb * self.Nd), dtype=np.float32),
            'tau': np.zeros((n_samples, self.L_max), dtype=np.float32),
            'f': np.zeros((n_samples, self.L_max), dtype=np.float32),
            'C_real': np.zeros((n_samples, self.L_max), dtype=np.float32),
            'C_imag': np.zeros((n_samples, self.L_max), dtype=np.float32),
            'L_true': np.zeros((n_samples,), dtype=np.int32),
            'sigma': np.zeros((n_samples,), dtype=np.float32),
            'phi_real': np.zeros((n_samples, self.Nb * self.Nd), dtype=np.float32),
            'phi_imag': np.zeros((n_samples, self.Nb * self.Nd), dtype=np.float32),
            'ser': np.zeros((n_samples,), dtype=np.float32)
        }

        for i in range(n_samples):
            if i % 1000 == 0:
                print(f"  {split_name}: 已生成 {i}/{n_samples} 样本")

            # # 随机生成目标数量 (1到L_max)
            # L = np.random.randint(1, self.L_max + 1)

            # 生成目标数量
            L = self.L_max

            # 生成单个样本
            sample = self._generate_single_sample(L)

            # 存储到数组
            data_dict['y_real'][i] = sample['y'].real
            data_dict['y_imag'][i] = sample['y'].imag
            data_dict['b_real'][i] = sample['b'].real
            data_dict['b_imag'][i] = sample['b'].imag
            data_dict['tau'][i, :L] = sample['tau']
            data_dict['f'][i, :L] = sample['f']
            data_dict['C_real'][i, :L] = sample['C'].real
            data_dict['C_imag'][i, :L] = sample['C'].imag
            data_dict['L_true'][i] = L
            data_dict['sigma'][i] = sample['sigma']
            data_dict['phi_real'][i] = sample['phi'].real
            data_dict['phi_imag'][i] = sample['phi'].imag
            data_dict['ser'][i] = sample['ser']

        return data_dict

    def _generate_single_sample(self, L: int) -> Dict:
        # 1. 生成随机目标参数
        tau = np.random.uniform(*self.tau_range, L)
        f = np.random.uniform(*self.f_range, L)

        # 2. 生成复反射系数
        C_real = np.random.normal(0, 0.7, L)
        C_imag = np.random.normal(0, 0.7, L)
        C = C_real + 1j * C_imag

        # 3. 生成S和D矩阵（VanderMonde结构）
        S = np.zeros((self.Nb, L), dtype=complex)
        D = np.zeros((self.Nd, L), dtype=complex)

        for i in range(L):
            S[:, i] = vander_vec(0, (self.Nb - 1) * f[i], self.Nb).reshape(-1)
            D[:, i] = vander_vec(0, (self.Nd - 1) * tau[i], self.Nd).reshape(-1)

        # 4. 构建Psi矩阵（Khatri-Rao积）
        Psi = kr(S, np.conj(D)) @ C.reshape(-1, 1)

        # 5. 生成通信符号（QPSK调制）
        b, e, ser = self._generate_communication_symbols()

        # 6. 生成观测信号y
        real_y = np.diag(b + e) @ Psi

        # 7. 添加噪声
        snr_w = np.random.uniform(*self.snr_range)
        w = np.sqrt(1 / 2) * (np.random.randn(self.Nb * self.Nd, 1) + 1j * np.random.randn(self.Nb * self.Nd, 1))
        w_var = np.linalg.norm(real_y) ** 2 / (10 ** (snr_w / 10) * self.Nb * self.Nd)

        y = real_y + np.sqrt(w_var) * w
        y = y.flatten()
        # ANM-DUMV参数
        lambda_val = 1
        sigma = np.linalg.norm(e / b) + 1
        opts = {
            'eta_abs': 1e-7,
            'eta_rel': 1e-7,
            'max_iter': 100
        }
        phi, _ = admm_for_us(y, b, self.Nd, self.Nb, lambda_val, sigma, opts)

        return {
            'y': y.astype(np.complex64),
            'b': b.astype(np.complex64),
            'tau': tau.astype(np.float32),
            'f': f.astype(np.float32),
            'C': C.astype(np.complex64),
            'sigma': np.float32(sigma),
            'ser': ser,
            'phi': phi.astype(np.complex64)
        }

    def create_pytorch_dataloader(self, batch_size=32, split='train', shuffle=True):
        """

        :param batch_size:
        :param split: 分片
        :param shuffle: 是否打乱
        :return: dataloader
        """
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        # 加载数据
        split_dir = self.data_dir / split
        if not split_dir.exists():
            raise ValueError(f"分割 {split} 不存在，请先生成数据集")

        # 加载所有npy文件
        data_arrays = {}
        for file in split_dir.glob('*.npy'):
            key = file.stem
            data_arrays[key] = np.load(file)

        # 将复数数据组合成复数张量
        y_real = torch.FloatTensor(data_arrays['y_real'])
        y_imag = torch.FloatTensor(data_arrays['y_imag'])
        y = torch.complex(y_real, y_imag)

        b_real = torch.FloatTensor(data_arrays['b_real'])
        b_imag = torch.FloatTensor(data_arrays['b_imag'])
        b = torch.complex(b_real, b_imag)

        # 其他数据
        tau = torch.FloatTensor(data_arrays['tau'])
        f = torch.FloatTensor(data_arrays['f'])
        C_real = torch.FloatTensor(data_arrays['C_real'])
        C_imag = torch.FloatTensor(data_arrays['C_imag'])
        C = torch.complex(C_real, C_imag)

        L_true = torch.LongTensor(data_arrays['L_true'])
        sigma = torch.FloatTensor(data_arrays['sigma'])

        phi_real = torch.FloatTensor(data_arrays['phi_real'])
        phi_imag = torch.FloatTensor(data_arrays['phi_imag'])
        phi = torch.complex(phi_real, phi_imag)

        # 创建TensorDataset
        dataset = TensorDataset(y, b, tau, f, C, L_true, sigma, phi)

        # 创建DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return dataloader
