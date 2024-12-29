import torch
import torch.nn as nn
import timm
import warnings
from torch.optim.lr_scheduler import _LRScheduler
import math
from typing import Any, Dict


class SqrtLoss(nn.Module):
    def __init__(self, beta=100, epsilon=1e-8):
        super(SqrtLoss, self).__init__()
        self.beta = beta
        self.epsilon = epsilon

    def forward(self, inputs, targets):
        """
        计算基于平方根的损失。

        Args:
            input (torch.Tensor): 模型的预测值。
            target (torch.Tensor): 真实值。

        Returns:
            torch.Tensor: 计算得到的损失值（标量）。
        """
        error = targets - inputs
        loss = self.beta * torch.pow(torch.abs(error) / self.beta + self.epsilon, 0.5)
        return loss.mean()


def get_outnorm(x: torch.Tensor, out_norm: str = "") -> torch.Tensor:
    """Common function to get a loss normalization value. Can
    normalize by either the batch size ('b'), the number of
    channels ('c'), the image size ('i') or combinations
    ('bi', 'bci', etc)
    """
    # b, c, h, w = x.size()
    img_shape = x.shape

    if not out_norm:
        return 1

    norm = 1
    if "b" in out_norm:
        # normalize by batch size
        # norm /= b
        norm /= img_shape[0]
    if "c" in out_norm:
        # normalize by the number of channels
        # norm /= c
        norm /= img_shape[-3]
    if "i" in out_norm:
        # normalize by image/map size
        # norm /= h*w
        norm /= img_shape[-1] * img_shape[-2]

    return norm


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6, out_norm: str = "bci"):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.out_norm = out_norm

    def forward(self, x, y):
        norm = get_outnorm(x, self.out_norm)
        loss = torch.sum(torch.sqrt((x - y).pow(2) + self.eps**2))
        return loss * norm


class LPIPS(nn.Module):
    def __init__(self, model_name, pretrained=False, weights=None):  # 示例权重
        super(LPIPS, self).__init__()
        self.feature_net = timm.create_model(
            model_name,  # swinv2_large_window12to16_192to256.ms_in22k_ft_in1k
            pretrained=pretrained,
            features_only=True,
        )
        if weights is None:
            channels_len = len(self.feature_net.feature_info.channels())
            weights = [(channels_len - i) * 0.1 for i in range(channels_len)]
        self.l1loss = nn.L1Loss()
        self.weights = weights
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, x, y):
        """
        计算 LPIPS 损失。

        Args:
            x: 输入图像，形状为 [B, C, H, W]
            y: 目标图像，形状为 [B, C, H, W]

        Returns:
            LPIPS 损失，一个标量。
        """
        f_x = self.feature_net(x)
        f_y = self.feature_net(y)
        loss = 0
        for i in range(len(f_x)):
            # 特征归一化
            f_x_norm = f_x[i] / (f_x[i].norm(dim=1, keepdim=True) + 1e-10)
            f_y_norm = f_y[i] / (f_y[i].norm(dim=1, keepdim=True) + 1e-10)
            loss += self.weights[i] * self.l1loss(f_x_norm, f_y_norm)
        return loss


class SemanticLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.semantic_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

    @torch.no_grad()
    def forward(self, x, y):
        f_x = self.semantic_model(x)
        f_y = self.semantic_model(y)
        return torch.nn.functional.l1_loss(f_x, f_y)


class CustomLRScheduler(_LRScheduler):
    def __init__(self, optimizer, total_steps, last_epoch=-1):
        self.epochs = total_steps
        # 定义里程碑
        self.milestone1 = int(total_steps * 0.01)
        self.milestone2 = int(total_steps * 0.66)
        self.milestone3 = int(total_steps * 0.95)
        self.milestone4 = int(total_steps * 0.98)

        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.milestone1:
            # 线性上升
            scale = (self.last_epoch + 1) / self.milestone1
            return [base_lr * scale for base_lr in self.base_lrs]
        elif self.last_epoch < self.milestone2:
            # 保持不变
            return self.base_lrs
        elif self.last_epoch < self.milestone3:
            # 指数衰减
            decay_epochs = self.milestone3 - self.milestone2
            current_decay_epoch = self.last_epoch - self.milestone2
            scale = 0.1 ** (current_decay_epoch / decay_epochs)
            return [base_lr * scale for base_lr in self.base_lrs]
        elif self.last_epoch < self.milestone4:
            return [base_lr * 0.1 for base_lr in self.base_lrs]
        else:
            return [base_lr * 0.1 / 3 for base_lr in self.base_lrs]


if __name__ == "__main__":
    x = torch.randn(2, 3, 256, 256)
    y = torch.randn(2, 3, 256, 256)
    loss = SemanticLoss()
    print(loss(x, y))
