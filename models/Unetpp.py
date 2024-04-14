import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from models.head import *


class CenterBlock(nn.Module):

    def __init__(self, head_channels, use_batchnorm) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(head_channels, head_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(head_channels, head_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(head_channels) if use_batchnorm else nn.Identity()
        self.bn2 = nn.BatchNorm2d(head_channels) if use_batchnorm else nn.Identity()
        self.act = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=False,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.act = nn.LeakyReLU(0.2, True)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        return x


class UnetPlusPlusDecoder(nn.Module):
    def __init__(
        self,
        n_blocks=4,
        encoder_channels=[64, 256, 512, 1536, 3072],
        decoder_channels=[256, 128, 64, 32],
        use_batchnorm=False,
        center=True,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )
        # encoder_channels = encoder_channels[1:]
        encoder_channels = encoder_channels[::-1]

        head_channels = encoder_channels[0]
        self.in_channels = [head_channels] + list(decoder_channels[:-1])
        self.skip_channels = list(encoder_channels[1:]) + [0]
        self.out_channels = decoder_channels

        if center:
            self.center = CenterBlock(head_channels, use_batchnorm)
        else:
            self.center = nn.Identity()

        blocks = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(layer_idx + 1):
                if depth_idx == 0:
                    in_ch = self.in_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1)
                    out_ch = self.out_channels[layer_idx]
                else:
                    out_ch = self.skip_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (
                        layer_idx + 1 - depth_idx
                    )
                    in_ch = self.skip_channels[layer_idx - 1]
                blocks[f"x_{depth_idx}_{layer_idx}"] = DecoderBlock(
                    in_ch, skip_ch, out_ch, use_batchnorm
                )

        blocks[f"x_{0}_{len(self.in_channels)-1}"] = DecoderBlock(
            self.in_channels[-1], 0, self.out_channels[-1], use_batchnorm
        )
        self.blocks = nn.ModuleDict(blocks)
        self.depth = len(self.in_channels) - 1

    def forward(self, *features):
        features = features[::-1]
        # start building dense connections
        dense_x = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(self.depth - layer_idx):
                if layer_idx == 0:
                    output = self.blocks[f"x_{depth_idx}_{depth_idx}"](
                        features[depth_idx], features[depth_idx + 1]
                    )
                    dense_x[f"x_{depth_idx}_{depth_idx}"] = output
                else:
                    dense_l_i = depth_idx + layer_idx
                    cat_features = [
                        dense_x[f"x_{idx}_{dense_l_i}"]
                        for idx in range(depth_idx + 1, dense_l_i + 1)
                    ]
                    cat_features = torch.cat(
                        cat_features + [features[dense_l_i + 1]], dim=1
                    )
                    dense_x[f"x_{depth_idx}_{dense_l_i}"] = self.blocks[
                        f"x_{depth_idx}_{dense_l_i}"
                    ](dense_x[f"x_{depth_idx}_{dense_l_i-1}"], cat_features)
        dense_x[f"x_{0}_{self.depth}"] = self.blocks[f"x_{0}_{self.depth}"](
            dense_x[f"x_{0}_{self.depth-1}"]
        )
        return dense_x[f"x_{0}_{self.depth}"]


class FullUnetPlusPlus(nn.Module):
    def __init__(
        self,
        n_blocks=4,
        encoder_channels=[128, 256, 512, 1024],
        decoder_channels=[512, 256, 128, 128],
        use_batchnorm=False,
        center=True,
    ):
        super().__init__()

        self.encoder = timm.create_model(
            "convnextv2_base.fcmae_ft_in22k_in1k",
            pretrained=True,
            features_only=True,
        )
        self.decoder = UnetPlusPlusDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            n_blocks=n_blocks,
            use_batchnorm=use_batchnorm,
            center=center,
        )
        self.up = nn.Sequential(nn.PixelShuffle(2), CenterBlock(32, use_batchnorm))
        self.head = mscheadv5(32)

    def forward(self, x):
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        decoder_output = self.up(decoder_output)
        pred = self.head(decoder_output)
        return pred


if __name__ == "__main__":
    model = FullUnetPlusPlus()
    x = torch.randn(2, 3, 384, 384)
    y = model(x)
    print(y.shape)
