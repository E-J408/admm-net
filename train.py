import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import time
from pathlib import Path

from admm_net import ADMMNet
from generate_data import OFDMDatasetGenerator
from loss import BasicANMLoss


def main(data_dir, train_name):
    # ------配置参数--------
    checkpoints_dir = 'checkpoints' + '/' + train_name
    logs_dir = 'logs' + '/' + train_name
    config = {
        # 数据参数
        'data_dir': data_dir,
        'batch_size': 32,
        'num_workers': 4,

        # 模型参数
        'num_layers': 10,  # ADMM展开的层数
        'M': 10,  # 符号数
        'N': 10,  # 子载波数
        'L_max': 3,  # 最大目标数
        'hidden_dim': 128,  # 隐藏层维度

        # 训练参数
        'epochs': 100,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',

        # 模型保存参数
        'checkpoint_dir': checkpoints_dir,
        'log_dir': logs_dir
    }
    # 创建目录
    Path(config['checkpoint_dir']).mkdir(exist_ok=True)
    Path(config['log_dir']).mkdir(exist_ok=True)

    # 保存配置
    with open(Path(config['log_dir']) / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"使用设备: {config['device']}")
    print("配置参数:")
    for key, value in config.items():
        print(f"  {key}: {value}")


    # -----创建数据集和数据加载器--------
    print("\n" + "=" * 50)
    print("加载数据集...")

    # 初始化数据生成器（假设已生成数据集）
    data_generator = OFDMDatasetGenerator(
        Nb=config['M'],
        Nd=config['N'],
        L_max=config['L_max'],
        data_dir=config['data_dir']
    )

    # 创建数据加载器
    train_loader = data_generator.create_pytorch_dataloader(
        batch_size=config['batch_size'],
        split='train',
        shuffle=True
    )
    val_loader = data_generator.create_pytorch_dataloader(
        batch_size=config['batch_size'],
        split='val',
        shuffle=False
    )
    test_loader = data_generator.create_pytorch_dataloader(
        batch_size=config['batch_size'],
        split='test',
        shuffle=False
    )

    print(f"训练集: {len(train_loader.dataset)} 样本")
    print(f"验证集: {len(val_loader.dataset)} 样本")
    print(f"测试集: {len(test_loader.dataset)} 样本")

    # -----创建模型--------
    print("\n" + "=" * 50)
    print("初始化模型...")

    model = ADMMNet(
        num_layers=config['num_layers'],
        M=config['M'],
        N=config['N'],
        L=config['L_max'],
    ).to(config['device'])

    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数总数: {total_params:,}")

    # -----创建损失函数和优化器--------
    criterion = BasicANMLoss()
    # 参数分组优化：不同组件使用不同学习率
    param_groups = []

    # ADMM层参数（相对稳定，使用较小学习率）
    admm_params = []
    for name, param in model.named_parameters():
        if any(prefix in name for prefix in ['phiLayers', 'hLayers', 'gLayers', 'zLayers']):
            admm_params.append(param)
    if admm_params:
        param_groups.append({'params': admm_params, 'lr': config['lr'] * 0.5})

        # 峰值搜索层参数（需要更灵活的学习）
    peak_params = []
    for name, param in model.named_parameters():
        if 'peakSearchLayer' in name:
            peak_params.append(param)
    if peak_params:
        param_groups.append({'params': peak_params, 'lr': config['lr']})

    # 其他参数
    other_params = []
    for name, param in model.named_parameters():
        if param not in admm_params and param not in peak_params:
            other_params.append(param)
    if other_params:
        param_groups.append({'params': other_params, 'lr': config['lr']})

    optimizer = optim.AdamW(param_groups, lr=config['lr'],
                            weight_decay=config['weight_decay'])
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )  # 采用余弦退火学习率调度器，T0意为循环次数，Tmult意为循环次数的乘数，eta_min意为最小学习率

    # -----训练准备--------
    start_epoch = 0
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    # 加载检查点（如果存在）
    checkpoint_path = Path(config['checkpoint_dir']) / 'best_model.pth'
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"加载检查点，从epoch {start_epoch}继续训练")

    # 记录器
    history = {
        'train_loss': [], 'val_loss': [],
        'tau_rmse': [], 'f_rmse': [],
        'learning_rate': []
    }

    # -----训练循环--------
    print("\n" + "=" * 50)
    print("开始训练...")

    for epoch in range(start_epoch, config['epochs']):
        start_time = time.time()

        # 训练阶段
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch_idx, batch_data in enumerate(train_loader):
            # 解包数据
            y, b, tau_true, f_true, C, L_true, sigma = batch_data
            y = y.to(config['device'])
            b = b.to(config['device'])
            tau_true = tau_true.to(config['device'])
            f_true = f_true.to(config['device'])
            L_true = L_true.to(config['device'])
            sigma = sigma.to(config['device'])

            # 前向传播
            optimizer.zero_grad()

            tau_est, f_est, confidences = model(y, b, sigma)

            # 转换为结果字典
            model_outputs = {
                'tau_est': tau_est,
                'f_est': f_est,
                'confidences': confidences
            }

            # 准备真实值字典
            ground_truth = {
                'tau_true': tau_true,
                'f_true': f_true,
                'L_true': L_true,
                'y': y,
                'b': b
            }

            total_loss, loss_dict = criterion(model_outputs, ground_truth)

            # 反向传播和优化
            total_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 参数更新
            optimizer.step()

            # 记录损失
            train_loss += total_loss.item()
            train_batches += 1

            # 打印进度
            if batch_idx % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Epoch: {epoch + 1}/{config["epochs"]} '
                      f'Batch: {batch_idx}/{len(train_loader)} '
                      f'Loss: {total_loss.item():.6f} '
                      f'LR: {current_lr:.2e}')









if __name__ == '__main__':
    # 训练名称
    train_name = time.strftime("%Y%m%d-%H-%M")
    data_dir = 'data/testDataGen'
    main(data_dir, train_name)