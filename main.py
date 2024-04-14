import torch
from option import get_option
from dataset import *

from pl_tool_gan import *

# from pl_tool import *
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
import torch
import wandb
import segmentation_models_pytorch as smp

torch.set_float32_matmul_precision("high")


if __name__ == "__main__":
    opt = get_option()
    """定义网络"""
    # model = smp.UnetPlusPlus(
    #     encoder_name="se_resnext101_32x4d",
    #     encoder_depth=5,
    #     encoder_weights="imagenet",
    #     decoder_channels=(512, 256, 128, 64, 32),
    #     decoder_attention_type=None,
    #     decoder_use_batchnorm=False,
    #     in_channels=3,
    #     classes=3,
    # )

    # model.segmentation_head = mscheadv5(32)

    from models.fusenet import fuse_convnext_swinv2, convnext_plus_head

    model = convnext_plus_head()
    # model = fuse_convnext_swinv2()

    """模型编译"""
    # model = torch.compile(model)

    """导入数据集"""
    train_dataloader, valid_dataloader = get_dataloader(opt)

    """Lightning 模块定义"""
    pl.seed_everything(opt.seed)
    wandb_logger = WandbLogger(
        project=opt.project,
        name=opt.exp_name,
        offline=not opt.save_wandb,
        config=opt,
    )

    trainer = pl.Trainer(
        accelerator="auto",
        devices=[opt.devices],
        strategy="auto",
        max_epochs=opt.epochs,
        # precision="bf16-mixed",
        default_root_dir="./",
        deterministic=False,
        logger=wandb_logger,
        val_check_interval=opt.val_check,
        log_every_n_steps=opt.log_step,
        accumulate_grad_batches=1,
        # gradient_clip_val=opt.grad_clip,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=f"./checkpoints/" + opt.exp_name,
                monitor="valid_psnr",
                mode="max",
                save_top_k=3,
                save_last=True,
                filename="{epoch}-{valid_psnr:.4f}",
            ),
        ],
    )

    # Start training
    trainer.fit(
        LightningModule(opt, model, len(train_dataloader)),
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )
    wandb.finish()
