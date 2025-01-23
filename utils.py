import torch
import torch.nn as nn
import timm


class SmoothFocalL1Loss(nn.Module):
    def __init__(
        self,
        gamma_small=0.5,
        gamma_large=2.0,
        threshold=1.0,
        beta=0.5,
        reduction="mean",
    ):
        super(SmoothFocalL1Loss, self).__init__()
        self.gamma_small = gamma_small
        self.gamma_large = gamma_large
        self.threshold = threshold
        self.beta = beta  # 控制过渡的平滑程度
        self.reduction = reduction

    def forward(self, input, target):
        l1_loss = torch.abs(input - target)

        # 使用 sigmoid 函数实现平滑过渡
        sigmoid = 1 / (1 + torch.exp(-self.beta * (l1_loss - self.threshold)))
        gamma = sigmoid * (self.gamma_large - self.gamma_small) + self.gamma_small

        loss = l1_loss * (l1_loss**gamma)

        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:
            raise NotImplementedError


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
    def __init__(self, lpips_net):
        super().__init__()
        self.semantic_model = torch.hub.load("facebookresearch/dinov2", lpips_net)

    @torch.no_grad()
    def forward(self, x, y):
        f_x = self.semantic_model(x)
        f_y = self.semantic_model(y)
        return torch.nn.functional.l1_loss(f_x, f_y)


class DINOv2DNet(nn.Module):
    def __init__(self, dnet_net):
        super().__init__()
        self.semantic_model = torch.hub.load("facebookresearch/dinov2", dnet_net)
        for param in self.semantic_model.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(1024, 1)

    def forward(self, image):
        f = self.semantic_model(image)
        pred = self.fc(f)
        return pred


if __name__ == "__main__":
    x = torch.randn(2, 3, 256, 256)
    y = torch.randn(2, 3, 256, 256)
    loss = SemanticLoss()
    print(loss(x, y))
