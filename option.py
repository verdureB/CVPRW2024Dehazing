import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save_wandb", type=bool, default=True)
    parser.add_argument("--project", type=str, default="Dehaze")

    # models

    # augmentations

    # dataset
    parser.add_argument("--dataset_root", type=str, default="./dehaze_data/")
    parser.add_argument("--image_size", type=int, default=256)

    # training setups
    parser.add_argument("-wd", "--weight_decay", type=float, default=1e-8)
    parser.add_argument("-lr", "--learning_rate", type=float, default=8e-5)
    parser.add_argument("-bs", "--batch_size", type=int, default=3)
    parser.add_argument("-e", "--epochs", type=int, default=2000)
    parser.add_argument("--num_workers", type=int, default=16)

    # experiment
    parser.add_argument("--devices", type=int, default=4)
    parser.add_argument("--exp_name", type=str, default="baseline_rdnetlpips")
    parser.add_argument("--val_check", type=float, default=1.0)
    parser.add_argument("--log_step", type=int, default=10)

    return parser.parse_args()


def get_option():
    opt = parse_args()
    return opt
