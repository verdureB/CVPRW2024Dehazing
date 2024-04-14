import torch
import lightning.pytorch as pl
import torch.nn.functional as F
from lion_pytorch import Lion
import numpy as np
from models.head import *
import torchmetrics as tm
from torch.optim import AdamW
from utils import CharbonnierLoss

torch.set_float32_matmul_precision("high")


class LightningModule(pl.LightningModule):
    def __init__(self, opt, model, len_trainloader):
        super().__init__()
        self.learning_rate = opt.learning_rate
        self.len_trainloader = len_trainloader
        self.opt = opt
        self.model = model
        self.l1loss = CharbonnierLoss()

        # ckpt = torch.load(
        #     "checkpoints/last.ckpt",
        #     map_location="cpu",
        # )["state_dict"]
        # for k in list(ckpt.keys()):
        #     if "model." in k:
        #         ckpt[k.replace("model.", "")] = ckpt.pop(k)
        # self.model.load_state_dict(ckpt, strict=False)
        # vgg_model = vgg16(pretrained=True)
        # vgg_model = vgg_model.features[:16]
        # for param in vgg_model.parameters():
        #     param.requires_grad = False

        # self.lpipsloss = LossNetwork(vgg_model)
        # self.lpipsloss.eval()

    def forward(self, x):
        pred = self.model(x)
        return pred

    def configure_optimizers(self):
        self.optimizer = AdamW(
            self.parameters(),
            weight_decay=self.opt.weight_decay,
            lr=self.learning_rate,
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            epochs=self.opt.epochs,
            pct_start=0.06,
            steps_per_epoch=self.len_trainloader,
        )
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",
            },
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        # pred1, pred2 = self.model(x)
        # pred = pred1 * x + x + pred2
        pred = self.model(x)
        pred = torch.clamp(pred, -1, 1)
        l1loss = self.l1loss(pred, y)
        # lpipsloss = self.lpipsloss(pred, y)
        # msssimloss = -msssim(pred, y)
        loss = l1loss
        psnr = tm.functional.image.peak_signal_noise_ratio(pred, y)
        ssim = tm.functional.image.structural_similarity_index_measure(pred, y)
        self.log("train_l1loss", l1loss)
        # self.log("train_lpipsloss", lpipsloss)
        # self.log("train_msssimloss", msssimloss)
        self.log("train_psnr", psnr)
        self.log("train_ssim", ssim)
        self.log("learning_rate", self.scheduler.get_last_lr()[0])
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        b, c, h, w = x.shape
        size = self.opt.image_size
        pred = torch.ones((b, c, h, w), device=x.device)
        m, n = h // size, w // size
        for i in range(m):
            for j in range(n):
                patch = x[:, :, i * size : (i + 1) * size, j * size : (j + 1) * size]
                patch = self.model(patch)
                # pred1, pred2 = self.model(patch)
                # patch = pred1 * patch + patch + pred2
                pred[:, :, i * size : (i + 1) * size, j * size : (j + 1) * size] = patch
        pred = torch.clamp(pred, -1, 1)
        l1loss = self.l1loss(pred, y)
        # lpipsloss = self.lpipsloss(pred, y)
        # msssimloss = -msssim(pred, y)
        psnr = tm.functional.image.peak_signal_noise_ratio(pred, y)
        ssim = tm.functional.image.structural_similarity_index_measure(pred, y)
        self.log("valid_psnr", psnr)
        self.log("valid_ssim", ssim)
        self.log("valid_l1loss", l1loss, prog_bar=True)
        # self.log("valid_lpipsloss", lpipsloss, prog_bar=True)
        # self.log("valid_msssimloss", msssimloss, prog_bar=True)

    def on_train_epoch_end(self):
        pass

    def on_validation_epoch_end(self):
        pass
