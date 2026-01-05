import torch
from pathlib import Path

from admm_net import PhiEstADMMNet
from generate_data import DatasetGeneratorCreatePhi
from loss import PhiAlignmentLoss
from utils.peakSearchUtils import *


def main(data_dir, train_name):
    # ------配置参数--------
    checkpoints_dir = '../checkpoints' + '/' + train_name
    logs_dir = '../logs' + '/' + train_name
    config = {
        # 数据参数
        'data_dir': data_dir,
        'batch_size': 256,
        'num_workers': 4,
        # 模型参数
        'num_layers': 10,  # ADMM展开的层数
        'M': 10,  # 符号数
        'N': 10,  # 子载波数
        'L_max': 3,  # 最大目标数
        'hidden_dim': 128,  # 隐藏层维度
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        # 模型保存参数
        'checkpoint_dir': checkpoints_dir,
        'log_dir': logs_dir
    }

    # 加载测试集
    print("\n" + "=" * 50)
    print("加载数据集...")
    # 初始化数据生成器（假设已生成数据集）
    data_generator = DatasetGeneratorCreatePhi(
        Nb=config['M'],
        Nd=config['N'],
        L_max=config['L_max'],
        data_dir=config['data_dir']
    )

    test_dataloader = data_generator.create_pytorch_dataloader(
        batch_size=config['batch_size'],
        split='test',
        shuffle=False
    )
    print(f"测试集: {len(test_dataloader.dataset)} 样本")

    # 加载最佳模型
    model = PhiEstADMMNet(
        num_layers=config['num_layers'],
        M=config['M'],
        N=config['N'],
        L=config['L_max'],
    ).to(config['device'])
    checkpoint = torch.load(Path(config['checkpoint_dir']) / 'best_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    # 加载损失函数
    criterion = PhiAlignmentLoss()

    model.eval()
    test_loss = 0.0
    test_batches = 0
    alt_peak_search_base = {
        'xbase': config['M'],
        'ybase': config['N'],
    }
    alt_peak_search_opts = {
        'xstep': 1 / (10 * config['M']),
        'ystep': 1 / (10 * config['N']),
        'iter': 3
    }
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_dataloader):
            y, b, tau_true, f_true, C, L_true, sigma, phi_true = batch_data
            y = y.to(config['device'])
            b = b.to(config['device'])
            sigma = sigma.to(config['device'])
            phi_true = phi_true.to(config['device'])

            # 前向传播
            phi = model(y, b, sigma)

            # 对每个phi值进行PeakSearch
            for i in range(phi.shape[0]):
                alt_peak_search_base['phi'] = phi[i].cpu().numpy()
                res = alt_peak_search(alt_peak_search_base, alt_peak_search_opts)
                # 对结果进行排序
                res = sorted(res, key=lambda x: x[2], reverse=True)
                # 取前L_true个峰值
                res = res[:L_true[i]]
                print(res)
                alt_peak_search_base['phi'] = phi_true[i].cpu().numpy()
                res_true = alt_peak_search(alt_peak_search_base, alt_peak_search_opts)
                res_true = sorted(res_true, key=lambda x: x[2], reverse=True)
                res_true = res_true[:L_true[i]]
                print(res_true)




            # 计算损失
            total_loss, loss_dict = criterion(phi, phi_true)
            test_loss += total_loss.item()
            test_batches += 1


    # 计算测试指标
    avg_test_loss = test_loss / test_batches if test_batches > 0 else 0

    print(f"\n测试结果:")
    print(f"  测试损失: {avg_test_loss:.6f}")


if __name__ == '__main__':
    cur_train_name = "20251226-10-47"
    cur_data_dir = '../data/phi_fixSNR20L3_1000'
    main(cur_data_dir, cur_train_name)