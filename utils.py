import torch
import torch.nn as nn
import timm


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
    def __init__(self, weights=[0.2, 0.4, 0.3, 0.1]):  # 示例权重
        super(LPIPS, self).__init__()
        self.feature_net = timm.create_model(
            "rdnet_base.nv_in1k", pretrained=True, features_only=True
        )
        self.l1loss = nn.L1Loss()  # 换成 L1 损失
        self.weights = weights  # 不同层的权重

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


if __name__ == "__main__":
    x = torch.randn(2, 3, 256, 256)
    y = torch.randn(2, 3, 256, 256)
    loss = lpips()
    print(loss(x, y))
