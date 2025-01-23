import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
import math
from .mstpp import MST_Plus_Plus


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        bias=False,
        norm=True,
        act=True,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(out_channels) if norm else None
        self.act = nn.ReLU() if act else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class MLA(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_dim,
        kernel_size=1,
        scales=(5,),
        dim=8,
        heads_ratio=1,
        heads=None,
        eps=1e-5,
    ):
        super(MLA, self).__init__()

        heads = int(in_channels // dim * heads_ratio) if heads is None else heads

        hidden_dim = heads * dim
        self.eps = eps
        self.qkv = ConvLayer(in_channels, hidden_dim * 3, kernel_size)
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        hidden_dim * 3, hidden_dim * 3, scale, 1, groups=hidden_dim * 3
                    ),
                    nn.Conv2d(hidden_dim * 3, hidden_dim * 3, 1, groups=hidden_dim * 3),
                )
                for scale in scales
            ]
        )
        self.act = nn.ReLU()
        self.proj = ConvLayer(hidden_dim * (1 + len(scales)), out_channels, 1)

    def relu_quadratic_attn(self, qkv):
        B, _, H, W = qkv.shape

        qkv = qkv.view(B, -1, 3 * self.dim, H * W)
        q, k, v = qkv.split(self.dim, dim=2)

        q = self.act(q)
        k = self.act(k)

        attn_map = torch.matmul(k.transpose(-1, -2), q)
        attn_map = attn_map / (torch.sum(attn_map, dim=2, keepdim=True) + self.eps)

        out = torch.matmul(v, attn_map)

        out = out.view(B, -1, H, W)
        return out

    def relu_linear_attn(self, qkv):
        pass

    def forward(self, x):
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        for aggreg in self.aggreg:
            multi_scale_qkv.append(aggreg(qkv))
        qkv = torch.cat(multi_scale_qkv, dim=1)

        H, W = x.shape[-2:]

        if H * W > self.dim:
            out = self.relu_linear_attn(qkv)
        else:
            out = self.relu_quadratic_attn(qkv)

        out = self.proj(out)


def default_conv(in_channels, out_channels, kernel_size, bias=False):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size // 2),
        bias=bias,
    )


class SKFusionv2(nn.Module):
    def __init__(self, height=2, kernel_size=9):
        super(SKFusionv2, self).__init__()

        self.height = height

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Conv1d(1, self.height, kernel_size, 1, kernel_size // 2)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.avg_pool(feats_sum)
        attn = attn.squeeze(-1).permute(0, 2, 1)
        attn = self.mlp(attn)
        attn = attn.permute(0, 2, 1)
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats * attn, dim=1)
        return out


class mscheadv5(nn.Module):
    def __init__(self, in_channels):
        super(mscheadv5, self).__init__()
        self.in_channels = in_channels
        self.head1 = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.head2 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.head3 = nn.Conv2d(in_channels, in_channels, 5, 1, 2)
        self.head4 = nn.Conv2d(in_channels, in_channels, 7, 1, 3)
        self.sk = SKFusionv2(height=4, kernel_size=7)
        self.b = nn.Sequential(nn.Conv2d(in_channels * 5, 3, 7, 1, 3), nn.Tanh())

    def forward(self, x):
        x1 = self.head1(x)
        x2 = self.head2(x)
        x3 = self.head3(x)
        x4 = self.head4(x)
        x = self.sk([x1, x2, x3, x4])
        x = torch.cat([x1, x2, x3, x4, x], dim=1)
        x = self.b(x)
        return x


class ConvNeXt(nn.Module):
    def __init__(
        self,
        block,
        in_chans=3,
        depths=[3, 3, 27],
        dims=[256, 512, 1024],
        drop_out_rate=0.0,
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
    ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dpath_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        dpout_rates = [x.item() for x in torch.linspace(0, drop_out_rate, sum(depths))]
        cur = 0
        for i in range(3):
            stage = nn.Sequential(
                *[
                    block(
                        dim=dims[i],
                        drop_out=dpout_rates[cur + j],
                        drop_path=dpath_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

    def forward(self, x):
        x_layer1 = self.downsample_layers[0](x)
        x_layer1 = self.stages[0](x_layer1)

        x_layer2 = self.downsample_layers[1](x_layer1)
        x_layer2 = self.stages[1](x_layer2)

        x_layer3 = self.downsample_layers[2](x_layer2)
        out = self.stages[2](x_layer3)

        return x_layer1, x_layer2, out


class Block(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_out=0.0, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_out = nn.Dropout(drop_out)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        inputs = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.drop_out(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = inputs + self.drop_path(x)
        return x


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class PALayer(nn.Module):
    def __init__(self, channel, bias=False):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=bias),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=bias),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel, bias=False):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, bias=bias),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel // 8, channel, 1, bias=bias),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class CP_Attention_block(nn.Module):
    def __init__(self, conv, dim, kernel_size, bias):
        super(CP_Attention_block, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=bias)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=bias)
        self.calayer = CALayer(dim, bias)
        self.palayer = PALayer(dim, bias)

    def forward(self, x):
        res = self.act1(self.conv1(x)) + x
        res = self.palayer(self.calayer(self.conv2(res))) + x
        return res


class knowledge_adaptation_convnext(nn.Module):
    def __init__(self, bias):
        super(knowledge_adaptation_convnext, self).__init__()
        self.encoder = ConvNeXt(
            Block,
            in_chans=3,
            depths=[3, 3, 27, 3],
            dims=[256, 512, 1024, 2048],
            drop_out_rate=0.2,
            drop_path_rate=0.2,
            layer_scale_init_value=1e-6,
        )

        checkpoint = torch.load("./models/convnext_xlarge_22k_1k_384_ema.pth")

        model_dict = self.encoder.state_dict()
        key_dict = {k: v for k, v in checkpoint["model"].items() if k in model_dict}
        model_dict.update(key_dict)
        self.encoder.load_state_dict(model_dict)
        del checkpoint
        del model_dict

        self.up_block = nn.PixelShuffle(2)
        self.attention0 = CP_Attention_block(default_conv, 1024, 3, bias)
        self.attention1 = CP_Attention_block(default_conv, 256, 3, bias)
        self.attention2 = CP_Attention_block(default_conv, 192, 3, bias)
        self.attention3 = CP_Attention_block(default_conv, 112, 5, bias)
        self.attention4 = CP_Attention_block(default_conv, 28, 5, bias)

    def forward(self, inputs):
        x_layer1, x_layer2, x_output = self.encoder(inputs)

        x_mid = self.attention0(x_output)  # [1024,24,24]

        x = self.up_block(x_mid)  # [256,48,48]
        x = self.attention1(x)

        x = torch.cat((x, x_layer2), 1)  # [768,48,48]

        x = self.up_block(x)  # [192,96,96]
        x = self.attention2(x)
        x = torch.cat((x, x_layer1), 1)  # [448,96,96]
        x = self.up_block(x)  # [112,192,192]
        x = self.attention3(x)

        x = self.up_block(x)  # [28,384,384]
        out = self.attention4(x)

        return out


class convnext_plus_head(nn.Module):
    def __init__(self, bias=False):
        super(convnext_plus_head, self).__init__()
        self.convnext_branch = knowledge_adaptation_convnext(bias=bias)
        # self.segmentation_head1 = mscheadv5(28)
        self.segmentation_head1 = nn.Sequential(nn.Conv2d(28, 3, 3, 1, 1), nn.Tanh())
        # self.segmentation_head2 = MST_Plus_Plus(3, 3, 30, 1)

    def forward(self, inputs):
        x_convnext = self.convnext_branch(inputs)
        pred = self.segmentation_head1(x_convnext)
        # pred = self.segmentation_head2(pred)
        return pred


if __name__ == "__main__":
    model = convnext_plus_head()
    inputs = torch.randn(2, 3, 256, 256)
    out = model(inputs)
    for o in out:
        print(o.size())
