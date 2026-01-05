import torch
import torch.optim as optim
import json
import time
from pathlib import Path

from admm_net import PhiEstADMMNet
from generate_data import DatasetGeneratorCreatePhi
from loss import PhiAlignmentLoss


def main(data_dir, train_name):
    # ------配置参数--------
    checkpoints_dir = 'checkpoints' + '/' + train_name
    logs_dir = 'logs' + '/' + train_name
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

        # 训练参数
        'epochs': 1000,
        'lr': 5e-3,
        'weight_decay': 1e-3,
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
    data_generator = DatasetGeneratorCreatePhi(
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

    model = PhiEstADMMNet(
        num_layers=config['num_layers'],
        M=config['M'],
        N=config['N'],
        L=config['L_max'],
    ).to(config['device'])

    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数总数: {total_params:,}")

    # -----创建损失函数和优化器--------
    criterion = PhiAlignmentLoss()
    # 参数分组优化：不同组件使用不同学习率
    param_groups = []

    # ADMM层参数（相对稳定，使用较小学习率）
    admm_params = []
    for name, param in model.named_parameters():
        if any(prefix in name for prefix in ['phiLayers', 'hLayers', 'gLayers', 'zLayers']):
            admm_params.append(param)
    if admm_params:
        param_groups.append({'params': admm_params, 'lr': config['lr'] * 0.5})

    optimizer = optim.AdamW(param_groups, lr=config['lr'],
                            weight_decay=config['weight_decay'])
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )  # 采用余弦退火学习率调度器，T0意为循环次数，T_mult意为循环次数的乘数，eta_min意为最小学习率

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
        'lr': []
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
            y, b, tau_true, f_true, C, L_true, sigma, phi_true = batch_data
            y = y.to(config['device'])
            b = b.to(config['device'])
            sigma = sigma.to(config['device'])
            phi_true = phi_true.to(config['device'])

            # 前向传播
            optimizer.zero_grad()

            phi = model(y, b, sigma)

            total_loss, loss_dict = criterion(phi, phi_true)

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

        # 计算训练损失
        avg_train_loss = train_loss / train_batches if train_batches > 0 else 0
        history['train_loss'].append(avg_train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(val_loader):
                y, b, tau_true, f_true, C, L_true, sigma, phi_true = batch_data
                y = y.to(config['device'])
                b = b.to(config['device'])
                sigma = sigma.to(config['device'])
                phi_true = phi_true.to(config['device'])

                # 前向传播
                phi = model(y, b, sigma)

                total_loss, loss_dict = criterion(phi, phi_true)
                val_loss += total_loss.item()
                val_batches += 1

        # 计算验证损失
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        history['val_loss'].append(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)

        epoch_time = time.time() - start_time

        # 打印epoch结果
        print(f"\nEpoch {epoch + 1}/{config['epochs']} - {epoch_time:.1f}s")
        print(f"  训练损失: {avg_train_loss:.6f}")
        print(f"  验证损失: {avg_val_loss:.6f}")
        print(f"  学习率: {current_lr:.2e}")

        # 学习率调度
        scheduler.step()

        # early stop和保存模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            # 保存模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'config': config,
                'history': history
            }, Path(config['checkpoint_dir']) / 'best_model.pth')
            print(f"  保存模型: {train_name}.pth 验证损失: {avg_val_loss:.6f}")
        else:
            patience_counter += 1
            print(f"  早停计数器: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"  早停: 训练结束")
            break
        # 保存训练历史
        with open(Path(config['log_dir']) / 'training_history.json', 'w') as f:
            # 转换为可JSON序列化的格式
            history_serializable = {k: [float(x) for x in v] for k, v in history.items()}
            json.dump(history_serializable, f, indent=2)

        print("-" * 50)

    # -------测试阶段--------
    print("\n" + "=" * 50)
    print("在测试集上评估最终模型...")

    # 加载最佳模型
    checkpoint = torch.load(Path(config['checkpoint_dir']) / 'best_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_loss = 0.0
    test_batches = 0
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            y, b, tau_true, f_true, C, L_true, sigma, phi_true = batch_data
            y = y.to(config['device'])
            b = b.to(config['device'])
            sigma = sigma.to(config['device'])
            phi_true = phi_true.to(config['device'])

            # 前向传播
            phi = model(y, b, sigma)

            # 计算损失
            total_loss, loss_dict = criterion(phi, phi_true)
            test_loss += total_loss.item()
            test_batches += 1
    # 计算测试指标
    avg_test_loss = test_loss / test_batches if test_batches > 0 else 0

    print(f"\n测试结果:")
    print(f"  测试损失: {avg_test_loss:.6f}")

    # 保存测试结果
    test_result = {
        'test_loss': avg_test_loss,
    }

    with open(Path(config['log_dir']) / 'test_result.json', 'w') as f:
        json.dump(test_result, f, indent=2)

    print("\n" + "=" * 50)
    print("训练结束")


if __name__ == '__main__':
    # 训练名称
    cur_train_name = time.strftime("%Y%m%d-%H-%M")
    cur_data_dir = 'data/phi_fixSNR20L3_1000_least5'
    main(cur_data_dir, cur_train_name)
