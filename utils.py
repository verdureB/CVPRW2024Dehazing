import torch
import torch.nn as nn
import torch.nn.functional as F


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


import timm


class swinv2_small_window8_256_lpips(nn.Module):
    def __init__(
        self,
    ):
        super(swinv2_small_window8_256_lpips, self).__init__()
        self.feature_net = timm.create_model(
            "swinv2_small_window8_256", pretrained=True, features_only=True
        )
        self.l1loss = CharbonnierLoss()

    @torch.no_grad()
    def forward(self, x, y):
        f_x = self.feature_net(x)
        f_y = self.feature_net(y)
        loss = 0
        for i in range(len(f_x)):
            loss += self.l1loss(f_x[i], f_y[i])
        return loss


if __name__ == "__main__":
    x = torch.randn(2, 3, 256, 256)
    y = torch.randn(2, 3, 256, 256)
    loss = CharbonnierLoss()
    print(loss(x, y))
