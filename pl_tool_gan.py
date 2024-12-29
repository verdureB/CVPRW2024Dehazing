import torch
import lightning.pytorch as pl
import torchmetrics as tm
from torch.optim import AdamW
import cv2
import numpy as np
from utils import CharbonnierLoss, LPIPS, SqrtLoss, SemanticLoss, CustomLRScheduler
import torchvision
from pytorch_msssim import msssim
import heavyball.utils as hu

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
        self.count = 0
        self.opt = opt
        self.model = model
        self.DNet = torchvision.models.densenet201(num_classes=1)
        self.lpips = SemanticLoss()
        self.l1loss = torch.nn.L1Loss()
        self.adversarial_loss = torch.nn.BCEWithLogitsLoss()
        self.automatic_optimization = False
        self.msssim_loss = msssim

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
        self.optimizer1 = AdamW(
            self.model.parameters(),
            weight_decay=self.opt.weight_decay,
            lr=self.learning_rate,
            betas=(0.9, self.opt.beta2),
        )
        self.optimizer2 = AdamW(
            self.DNet.parameters(),
            weight_decay=self.opt.weight_decay,
            lr=self.learning_rate,
            betas=(0.9, self.opt.beta2),
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer1,
            max_lr=self.learning_rate,
            epochs=self.opt.epochs,
            steps_per_epoch=self.len_trainloader,
            pct_start=self.opt.pct_start,
            anneal_strategy=self.opt.decay_mode,
        )
        return (
            {
                "optimizer": self.optimizer1,
                "lr_scheduler": {
                    "scheduler": self.scheduler,
                    "interval": "step",
                },
            },
            {"optimizer": self.optimizer2},
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        optimizer_g, optimizer_d = self.optimizers()

        # Train Generator
        self.toggle_optimizer(optimizer_g)

        pred = self.model(x)
        pred = torch.clamp(pred, -1, 1)
        l1loss = self.l1loss(pred, y)
        g_loss = (
            self.adversarial_loss(self.DNet(pred), self.valid)
            if self.opt.gan_g_rate > 0
            else 0
        )
        msssim_loss = (
            -self.msssim_loss(pred, y, normalize=True)
            if self.opt.msssim_rate > 0
            else 0
        )
        lpips_loss = self.lpips(pred, y) if self.opt.lpips_rate > 0 else 0

        loss = l1loss
        loss += self.opt.msssim_rate * msssim_loss
        loss += self.opt.lpips_rate * lpips_loss
        loss += self.opt.gan_g_rate * g_loss

        self.manual_backward(loss)
        optimizer_g.step()
        if self.scheduler is not None:
            scheduler = self.scheduler
            scheduler.step()
        optimizer_g.zero_grad()

        # Update EMA
        if self.ema_enabled and self.ema is not None:
            self.ema.update()

        self.untoggle_optimizer(optimizer_g)

        # Train Discriminator
        if self.opt.gan_d_rate > 0:
            self.toggle_optimizer(optimizer_d)

            d_loss = self.adversarial_loss(
                self.DNet(y), self.valid
            ) + self.adversarial_loss(self.DNet(pred.detach()), self.fake)
            d_loss = self.opt.gan_d_rate * d_loss / 2

            self.manual_backward(d_loss)
            optimizer_d.step()
            optimizer_d.zero_grad()

            self.untoggle_optimizer(optimizer_d)
        else:
            d_loss = 0

        # Logging Metrics
        psnr = tm.functional.image.peak_signal_noise_ratio(pred, y, data_range=2)
        ssim = tm.functional.image.structural_similarity_index_measure(
            pred, y, data_range=2
        )

        self.log("g_loss", g_loss)
        self.log("d_loss", d_loss)
        self.log("train_l1loss", l1loss)
        self.log("train_msssim_loss", msssim_loss)
        self.log("train_lpips_loss", lpips_loss)
        self.log("train_psnr", psnr)
        self.log("train_ssim", ssim)
        self.log("learning_rate", self.optimizer1.param_groups[0]["lr"])

    def validation_step(self, batch, batch_idx):
        if self.ema_enabled and self.ema is not None:
            self.ema.apply_shadow()

        x, y = batch
        b, c, h, w = x.shape
        size = self.opt.valid_patch_size
        stride = size // 2

        pred = torch.zeros((b, c, h, w), device=x.device)
        counts = torch.zeros((b, c, h, w), device=x.device, dtype=torch.uint8)

        m, n = (h - size) // stride + 1, (w - size) // stride + 1

        for i in range(m):
            for j in range(n):
                start_h = i * stride
                start_w = j * stride
                end_h = start_h + size
                end_w = start_w + size

                patch = x[:, :, start_h:end_h, start_w:end_w]
                patch_pred = self.model(patch)

                pred[:, :, start_h:end_h, start_w:end_w] += patch_pred
                counts[:, :, start_h:end_h, start_w:end_w] += 1

        pred /= counts

        pred = torch.clamp(pred, -1, 1)
        if self.current_epoch % 3 == 0:
            cv2.imwrite(
                "./checkpoints/"
                + self.opt.exp_name
                + f"/training_image/{self.current_epoch:04d}.jpg",
                cv2.resize(
                    (127.5 * pred[0].permute(1, 2, 0).cpu().numpy() + 127.5).astype(
                        np.uint8
                    ),
                    (0, 0),
                    fx=0.25,
                    fy=0.25,
                ),
            )
            self.count += 1

        l1loss = self.l1loss(pred, y)
        psnr = tm.functional.image.peak_signal_noise_ratio(pred, y, data_range=2)
        ssim = tm.functional.image.structural_similarity_index_measure(
            pred, y, data_range=2
        )

        if self.ema_enabled and self.ema is not None:
            self.ema.restore()

        self.log("valid_psnr", psnr)
        self.log("valid_ssim", ssim)
        self.log("valid_l1loss", l1loss, prog_bar=True)

    def on_train_epoch_end(self):
        pass

    def on_validation_epoch_end(self):
        pass
