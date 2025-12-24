import torch
import torch.nn as nn
import torch.nn.functional as F


def basic_parameter_loss(tau_pred, f_pred, tau_true, f_true, confidences, L_true):
    """
    基础参数估计损失
    """
    batch_size, L_max = tau_pred.shape
    total_loss = 0.0

    for b in range(batch_size):
        L = L_true[b]  # 该样本的真实目标数量
        if L == 0:
            # 若无目标，则惩罚所有预测的置信度，鼓励其趋近于0
            loss = torch.sum(confidences[b] ** 2)
        else:
            # 若有目标，计算前L个预测与真实值的MSE
            tau_loss = F.mse_loss(tau_pred[b, :L], tau_true[b, :L])
            f_loss = F.mse_loss(f_pred[b, :L], f_true[b, :L])
            # 对存在目标的预测，鼓励其置信度趋近于1
            # confidence_loss = F.mse_loss(confidences[b, :L], torch.ones(L, device=confidences.device))

            # loss = tau_loss + f_loss + 0.1 * confidence_loss
            loss = tau_loss + f_loss

        total_loss += loss

    return total_loss / batch_size


class BasicANMLoss(nn.Module):
    def __init__(self, lambda_reg=1e-4):
        super().__init__()
        self.lambda_reg = lambda_reg

    def forward(self, model_outputs, ground_truth):
        tau_pred = model_outputs['tau_est']
        f_pred = model_outputs['f_est']
        confidences = model_outputs['confidences']
        phi_final = model_outputs['phi_final']

        tau_true = ground_truth['tau_true']
        f_true = ground_truth['f_true']
        L_true = ground_truth['L_true']

        # 1. 核心参数损失
        param_loss = basic_parameter_loss(tau_pred, f_pred, tau_true, f_true, confidences, L_true)

        # 2. 轻量正则化（可选，用于稳定训练）
        # reg_loss = self.lambda_reg * torch.mean(torch.norm(phi_final, dim=1))

        # total_loss = param_loss + reg_loss
        total_loss = param_loss

        # loss_dict = {'total_loss': total_loss, 'param_loss': param_loss, 'reg_loss': reg_loss}
        loss_dict = {'total_loss': total_loss, 'param_loss': param_loss}

        return total_loss, loss_dict


