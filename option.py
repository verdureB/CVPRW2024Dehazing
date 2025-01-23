import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="用于图像去雾任务的训练脚本")

    # 数据集配置
    data_group = parser.add_argument_group("数据集配置", "与数据加载和预处理相关的参数")
    data_group.add_argument(
        "--dataset_root", type=str, default="./dehaze_data_1/", help="数据集根目录"
    )
    data_group.add_argument(
        "--image_size", type=int, default=336, help="训练时裁剪的图像尺寸"
    )
    data_group.add_argument(
        "--valid_image_size", type=int, default=4096, help="验证时使用的完整图像尺寸"
    )
    data_group.add_argument(
        "--valid_patch_size", type=int, default=2048, help="验证时裁剪的图像块尺寸"
    )
    data_group.add_argument(
        "--ori_image_rate", type=float, default=0.0, help="使用原始清晰图像的概率"
    )
    data_group.add_argument(
        "--valid_image_rate", type=float, default=0.12, help="作为验证集的百分比"
    )
    data_group.add_argument(
        "--crops_per_image", type=int, default=6, help="每张图像裁剪的次数"
    )
    data_group.add_argument(
        "--extra_data", help="是否使用额外数据集", action="store_true"
    )

    # 训练设置
    training_group = parser.add_argument_group(
        "训练设置", "控制模型训练过程的参数，例如优化器和学习率"
    )
    training_group.add_argument(
        "-wd", "--weight_decay", type=float, default=1e-8, help="权重衰减率"
    )
    training_group.add_argument(
        "-lr", "--learning_rate", type=float, default=1e-4, help="初始学习率"
    )
    training_group.add_argument(
        "-bs", "--batch_size", type=int, default=8, help="批量大小"
    )
    training_group.add_argument(
        "-e", "--epochs", type=int, default=5000, help="训练总轮数"
    )
    training_group.add_argument(
        "--pct_start", type=float, default=0.06, help="学习率预热的比例"
    )
    training_group.add_argument(
        "--decay_mode", type=str, default="cos", help="学习率衰减模式 (cos, linear 等)"
    )
    training_group.add_argument(
        "--num_workers", type=int, default=16, help="数据加载器使用的线程数"
    )
    training_group.add_argument(
        "--beta2", type=float, default=0.95, help="AdamW 优化器的 beta2 参数"
    )

    # 损失函数配置
    loss_group = parser.add_argument_group(
        "损失函数配置", "与损失函数及其权重相关的参数"
    )
    loss_group.add_argument(
        "--msssim_rate", type=float, default=0.0, help="MS-SSIM 损失的权重"
    )
    loss_group.add_argument(
        "--lpips_rate", type=float, default=0.1, help="LPIPS 损失的权重"
    )
    loss_group.add_argument(
        "--gan_g_rate", type=float, default=0.001, help="生成器的 GAN 损失权重"
    )
    loss_group.add_argument(
        "--gan_d_rate", type=float, default=0.0001, help="判别器的 GAN 损失权重"
    )
    loss_group.add_argument(
        "--dnet_net", type=str, default="dinov2_vits14", help="判别器的 GAN 网络类型"
    )
    loss_group.add_argument(
        "--lpips_net", type=str, default="dinov2_vits14", help="使用的 LPIPS 网络类型"
    )
    loss_group.add_argument(
        "--lpips_weight", type=list, default=None, help="LPIPS 损失的权重列表"
    )

    # 实验配置
    experiment_group = parser.add_argument_group(
        "实验配置", "与实验环境和记录相关的参数"
    )
    experiment_group.add_argument(
        "--seed", type=int, default=42, help="随机种子，用于结果复现"
    )
    experiment_group.add_argument(
        "--save_wandb", type=bool, default=True, help="是否保存到 WandB"
    )
    experiment_group.add_argument(
        "--project", type=str, default="Dehaze", help="WandB 项目名称"
    )
    experiment_group.add_argument(
        "-d", "--devices", type=int, default=5, help="使用的 GPU 设备 ID"
    )
    experiment_group.add_argument(
        "--exp_name",
        type=str,
        default="v3->cautiou+dpath0.2+dropout0.2+extra_data+cc",
        help="实验名称",
    )
    experiment_group.add_argument(
        "--val_check", type=float, default=1.0, help="验证频率 (多少个 epoch 验证一次)"
    )
    experiment_group.add_argument(
        "--log_step", type=int, default=25, help="日志记录频率 (多少个 batch 记录一次)"
    )

    return parser.parse_args()


def get_option():
    opt = parse_args()
    print("----- Options -----")
    max_len = max(len(key) for key in vars(opt).keys())
    for key, value in vars(opt).items():
        print(f"{key:<{max_len}} : {value}")
    print("----- End -----")
    return opt


if __name__ == "__main__":
    options = get_option()
