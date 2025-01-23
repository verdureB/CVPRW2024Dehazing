import torch
import lightning.pytorch as pl
import torchmetrics as tm
import heavyball
import cv2
import os
import numpy as np
from utils import *
import torchvision
from pytorch_msssim import msssim
import heavyball.utils as hu
import torch.nn.functional as F

hu.set_torch()
torch.set_float32_matmul_precision("high")


class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(param.device)

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # 如果设备不匹配，将shadow移动到正确的设备
                if self.shadow[name].device != param.device:
                    self.shadow[name] = self.shadow[name].to(param.device)
                new_average = (
                    self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                )
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # 确保设备匹配
                if self.shadow[name].device != param.device:
                    self.shadow[name] = self.shadow[name].to(param.device)
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]


class LightningModule(pl.LightningModule):
    def __init__(self, opt, model, len_trainloader):
        super().__init__()
        self.learning_rate = opt.learning_rate
        self.len_trainloader = len_trainloader
        self.opt = opt
        self.model = model
        self.DNet = torchvision.models.densenet201(num_classes=1)
        # self.DNet = DINOv2DNet(opt.dnet_net)
        self.lpips = SemanticLoss(opt.lpips_net)
        self.l1loss = torch.nn.L1Loss()
        self.adversarial_loss = torch.nn.BCEWithLogitsLoss()
        self.automatic_optimization = False
        self.msssim_loss = msssim
        self.valid_images = []
        self.max_valid_images = 9  # 存储的最大图像数量

        self.register_buffer("valid", torch.ones((opt.batch_size, 1)))
        self.register_buffer("fake", torch.zeros((opt.batch_size, 1)))

        # Initialize EMA after model is moved to correct device
        self.ema = None
        self.ema_enabled = False

    def setup(self, stage):
        if self.ema is None:
            self.ema = EMA(self.model, decay=0.9999)

    def forward(self, x):
        pred = self.model(x)
        return pred

    def configure_optimizers(self):
        self.optimizer1 = heavyball.ForeachAdamW(
            self.model.parameters(),
            weight_decay=self.opt.weight_decay,
            lr=self.learning_rate,
            betas=(0.9, self.opt.beta2),
            caution=True,
        )
        self.optimizer2 = heavyball.ForeachAdamW(
            self.DNet.parameters(),
            weight_decay=self.opt.weight_decay,
            lr=self.learning_rate,
            betas=(0.9, self.opt.beta2),
            caution=True,
        )

        self.scheduler1 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer1, T_0=self.len_trainloader * 2
        )
        self.scheduler2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer2, T_0=self.len_trainloader * 2
        )

        return (
            {
                "optimizer": self.optimizer1,
                "lr_scheduler": {
                    "scheduler": self.scheduler1,
                    "interval": "step",
                },
            },
            {
                "optimizer": self.optimizer2,
                "lr_scheduler": {
                    "scheduler": self.scheduler2,
                    "interval": "step",
                },
            },
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        optimizer_g, optimizer_d = self.optimizers()

        # Train Generator
        g_loss, pred = self._train_generator(x, y, optimizer_g)

        # Train Discriminator
        d_loss = (
            self._train_discriminator(pred, y, optimizer_d)
            if self.opt.gan_d_rate > 0
            else 0
        )

        # Calculate and log metrics
        psnr = tm.functional.image.peak_signal_noise_ratio(pred, y, data_range=2)
        ssim = tm.functional.image.structural_similarity_index_measure(
            pred, y, data_range=2
        )

        # Log all metrics
        self.log("d_loss", d_loss)
        self.log("g_loss", g_loss["gan"])
        self.log("train_l1loss", g_loss["l1"])
        self.log("train_msssim_loss", g_loss["msssim"])
        self.log("train_lpips_loss", g_loss["lpips"])
        self.log("train_psnr", psnr)
        self.log("train_ssim", ssim)
        self.log("learning_rate", self.optimizer1.param_groups[0]["lr"])

    def _train_generator(self, x, y, optimizer_g):
        """训练生成器"""
        self.toggle_optimizer(optimizer_g)

        # Forward pass
        pred = self.model(x)
        pred = torch.clamp(pred, -1, 1)

        # Calculate losses
        losses = {
            "l1": self.l1loss(pred, y),
            "gan": (
                self.adversarial_loss(self.DNet(pred), self.valid)
                if self.opt.gan_g_rate > 0
                else 0
            ),
            "msssim": (
                -self.msssim_loss(pred, y, normalize=True)
                if self.opt.msssim_rate > 0
                else 0
            ),
            "lpips": self.lpips(pred, y) if self.opt.lpips_rate > 0 else 0,
        }

        # Calculate total loss
        total_loss = losses["l1"]
        total_loss += self.opt.msssim_rate * losses["msssim"]
        total_loss += self.opt.lpips_rate * losses["lpips"]
        total_loss += self.opt.gan_g_rate * losses["gan"]
        losses["total"] = total_loss

        # Backward pass and optimization
        self.manual_backward(total_loss)
        optimizer_g.step()

        if self.scheduler1 is not None:
            self.scheduler1.step()
        optimizer_g.zero_grad()

        if self.ema_enabled and self.ema is not None:
            self.ema.update()

        self.untoggle_optimizer(optimizer_g)

        return losses, pred

    def _train_discriminator(self, pred, y, optimizer_d):
        """训练判别器"""
        self.toggle_optimizer(optimizer_d)

        d_loss = (
            self.adversarial_loss(self.DNet(y), self.valid)
            + self.adversarial_loss(self.DNet(pred.detach()), self.fake)
        ) / 2
        d_loss = self.opt.gan_d_rate * d_loss

        self.manual_backward(d_loss)
        optimizer_d.step()

        if self.scheduler2 is not None:
            self.scheduler2.step()
        optimizer_d.zero_grad()

        self.untoggle_optimizer(optimizer_d)

        return d_loss

    def validation_step(self, batch, batch_idx):
        if self.ema_enabled and self.ema is not None:
            self.ema.apply_shadow()
        x, y = batch
        b, c, h, w = x.shape
        size = self.opt.valid_patch_size  # 2048
        stride = 1536  # size // 4

        # 计算需要的padding大小
        pad_h = (size - h % stride) % stride
        pad_w = (size - w % stride) % stride

        # 对图像进行padding
        x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        b, c, h_pad, w_pad = x_padded.shape

        # 创建输出tensor
        pred_padded = torch.zeros((b, c, h_pad, w_pad), device=x.device)
        counts = torch.zeros((b, c, h_pad, w_pad), device=x.device, dtype=torch.uint8)

        # 计算滑窗次数
        m = (h_pad - size) // stride + 1
        n = (w_pad - size) // stride + 1

        # 滑窗推理
        for i in range(m):
            for j in range(n):
                start_h = i * stride
                start_w = j * stride
                end_h = start_h + size
                end_w = start_w + size

                patch = x_padded[:, :, start_h:end_h, start_w:end_w]
                patch_pred = self.model(patch)

                pred_padded[:, :, start_h:end_h, start_w:end_w] += patch_pred
                counts[:, :, start_h:end_h, start_w:end_w] += 1

        # 处理重叠区域的平均值
        pred_padded = torch.where(counts > 0, pred_padded / counts, pred_padded)
        pred_padded = torch.clamp(pred_padded, -1, 1)

        # 裁剪回原始大小
        pred = pred_padded[:, :, :h, :w]
        # 存储预测结果用于后续拼接
        if (
            self.current_epoch % 4 == 0
            and len(self.valid_images) < self.max_valid_images
        ):
            img = (127.5 * pred[0].permute(1, 2, 0).cpu().numpy() + 127.5).astype(
                np.uint8
            )
            self.valid_images.append(img)

        l1loss = self.l1loss(pred, y)
        psnr = tm.functional.image.peak_signal_noise_ratio(pred, y, data_range=2)
        ssim = tm.functional.image.structural_similarity_index_measure(
            pred, y, data_range=2
        )

        if self.ema_enabled and self.ema is not None:
            self.ema.restore()

        self.log("valid_psnr", psnr, prog_bar=True)
        self.log("valid_ssim", ssim)
        self.log("valid_l1loss", l1loss)

    def on_validation_epoch_end(self):
        # 在验证epoch结束时处理图像拼接
        if self.current_epoch % 4 == 0 and len(self.valid_images) > 0:
            grid_size = 3
            num_images = min(len(self.valid_images), grid_size * grid_size)

            if num_images > 0:
                # 获取单个图像的尺寸
                single_height, single_width = self.valid_images[0].shape[:2]

                # 创建空白画布
                grid_image = np.zeros(
                    (single_height * grid_size, single_width * grid_size, 3),
                    dtype=np.uint8,
                )

                # 填充图像
                for idx in range(num_images):
                    i = idx // grid_size  # 行索引
                    j = idx % grid_size  # 列索引

                    # 填充到对应位置
                    grid_image[
                        i * single_height : (i + 1) * single_height,
                        j * single_width : (j + 1) * single_width,
                    ] = self.valid_images[idx]

                # 调整最终图像大小
                final_image = cv2.resize(grid_image, (0, 0), fx=0.5, fy=0.5)

                # 保存图像
                save_path = os.path.join(
                    "./checkpoints",
                    self.opt.exp_name,
                    "training_image",
                    f"{self.current_epoch:04d}.jpg",
                )

                # 确保保存目录存在
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, final_image)

            # 清空存储的图像列表，为下一个验证epoch做准备
            self.valid_images = []
