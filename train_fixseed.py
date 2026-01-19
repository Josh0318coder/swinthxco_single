# train_fixab.py

import os
import sys
import wandb
import argparse
from tqdm import tqdm
from datetime import datetime
from zoneinfo import ZoneInfo
from time import gmtime, strftime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset, DataLoader
import torchvision.transforms as torch_transforms

from src.losses import (
    ContextualLoss_forward,
    Perceptual_loss,
    discriminator_loss_fn,
    generator_loss_fn,
    l1_loss_fn,
    smoothness_loss_fn,
)
from src.models.CNN.GAN_models import Discriminator_x64_224
from src.models.CNN.ColorVidNet import ColorVidNet
from src.models.CNN.FrameColor import frame_colorization
from src.models.CNN.NonlocalNet import WeightedAverage_color, NonlocalWeightedAverage, WarpNet
from src.models.vit.embed import SwinModel
from src.data.dataloader import VideosDataset, VideosDataset_ImageNet
from src.utils import CenterPad_threshold
from src.utils import (
    RGB2Lab,
    ToTensor,
    Normalize,
    LossHandler,
    uncenter_l,
    tensor_lab2rgb,
    print_num_params
)
from src.models.vit.utils import load_params
from src.scheduler import PolynomialLR

parser = argparse.ArgumentParser()
parser.add_argument("--video_data_root_list", type=str)
parser.add_argument("--data_root_imagenet", type=str)
parser.add_argument("--annotation_file_path_list", type=str)
parser.add_argument("--imagenet_pairs_file", type=str)
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--accumulation_steps", type=int, default=2)
parser.add_argument("--image_size", nargs='+', type=int, default=[224, 224])
parser.add_argument("--ic", type=int, default=4)
parser.add_argument("--epoch", type=int, default=40)
parser.add_argument("--resume", action='store_true')
parser.add_argument("--resume_checkpoint_dir", type=str, default="")
parser.add_argument("--pretrained_model", default='swinv2-cr-t-224', type=str)
parser.add_argument("--load_pretrained_model", action='store_true')
parser.add_argument("--pretrained_model_dir", type=str, default='ckpt')
parser.add_argument("--lr", type=float, default=3e-5)
parser.add_argument("--lr_step", type=int, default=1)
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
parser.add_argument("--checkpoint_step", type=int, default=500)
parser.add_argument("--save_interval_steps", type=int, default=5000,
                    help="Save checkpoint every N steps within epoch (0=disabled, overwrites epoch_X_latest/)")
parser.add_argument("--real_reference_probability", type=float, default=0.0)
parser.add_argument("--domain_invariant", action='store_true')
parser.add_argument("--weight_l1", type=float, default=1.0)
parser.add_argument("--weight_contextual", type=float, default=0.0)
parser.add_argument("--weight_perceptual", type=float, default=0.15)
parser.add_argument("--weight_smoothness", type=float, default=150.0)
parser.add_argument("--weight_gan", type=float, default=0.015)
parser.add_argument("--weight_nonlocal_smoothness", type=float, default=0.0)
parser.add_argument("--luminance_noise", type=float, default=0.0)
parser.add_argument("--permute_data", action='store_true')
parser.add_argument("--epoch_train_discriminator", type=int, default=3)
parser.add_argument("--use_wandb", action='store_true')
parser.add_argument("--wandb_token", type=str, default="")
parser.add_argument("--wandb_name", type=str, default="")

# Data Augmentation (Geometric)
parser.add_argument('--use_augmentation', action='store_true',
                    help='Enable reference image geometric augmentation')
parser.add_argument('--use_crop', action='store_true',
                    help='Enable RandomResizedCrop augmentation')
parser.add_argument('--use_flip', action='store_true', default=True,
                    help='Enable horizontal flip augmentation (default: True)')
parser.add_argument('--use_rotation', action='store_true', default=True,
                    help='Enable rotation augmentation (default: True)')
parser.add_argument('--use_tps', action='store_true', default=True,
                    help='Enable TPS augmentation (default: True)')
parser.add_argument('--crop_prob', type=float, default=0.5,
                    help='Probability of RandomResizedCrop (default: 0.5)')
parser.add_argument('--crop_scale_min', type=float, default=0.8,
                    help='Min scale for RandomResizedCrop (default: 0.8)')
parser.add_argument('--crop_scale_max', type=float, default=1.0,
                    help='Max scale for RandomResizedCrop (default: 1.0)')
parser.add_argument('--crop_ratio_min', type=float, default=0.9,
                    help='Min aspect ratio for RandomResizedCrop (default: 0.9)')
parser.add_argument('--crop_ratio_max', type=float, default=1.1,
                    help='Max aspect ratio for RandomResizedCrop (default: 1.1)')
parser.add_argument('--flip_prob', type=float, default=0.5,
                    help='Probability of horizontal flip (default: 0.5)')
parser.add_argument('--rotation_prob', type=float, default=0.3,
                    help='Probability of rotation (default: 0.3)')
parser.add_argument('--rotation_degrees', type=float, default=10,
                    help='Max rotation angle in degrees (default: 10)')
parser.add_argument('--tps_prob', type=float, default=0.7,
                    help='Probability of TPS transformation (default: 0.7)')
parser.add_argument('--tps_displacement', type=float, default=0.1,
                    help='TPS displacement range as ratio of image size (default: 0.1 = ±10%)')
parser.add_argument('--tps_control_points', type=int, default=5,
                    help='TPS control points grid size (default: 5 → 5×5=25 points)')

def uncenter_ab(ab):
    return ab * 127.0

def prepare_dataloader_simple(dataset, batch_size=4, pin_memory=False, num_workers=0):
    """簡單的數據載入器，無DDP支持"""
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=pin_memory,
                            num_workers=num_workers)
    return dataloader

def load_data():
    transforms_video = [
        torch_transforms.Resize(opt.image_size),
        RGB2Lab(),
        ToTensor(),
        Normalize(),
    ]

    # Initialize geometric augmentation (if enabled)
    ref_augmentation = None
    if opt.use_augmentation:
        from augmentation import GeometricAugmentation
        ref_augmentation = GeometricAugmentation(
            use_crop=opt.use_crop,
            use_flip=opt.use_flip,
            use_rotation=opt.use_rotation,
            use_tps=opt.use_tps,
            crop_prob=opt.crop_prob,
            crop_scale=(opt.crop_scale_min, opt.crop_scale_max),
            crop_ratio=(opt.crop_ratio_min, opt.crop_ratio_max),
            flip_prob=opt.flip_prob,
            rotation_prob=opt.rotation_prob,
            rotation_degrees=opt.rotation_degrees,
            tps_prob=opt.tps_prob,
            tps_displacement=opt.tps_displacement,
            tps_control_points=opt.tps_control_points
        )
        print(f"✅ Geometric augmentation enabled:")
        print(f"   - RandomResizedCrop: {opt.use_crop} (prob={opt.crop_prob}, scale={opt.crop_scale_min}-{opt.crop_scale_max}, ratio={opt.crop_ratio_min}-{opt.crop_ratio_max})")
        print(f"   - Flip: {opt.use_flip} (prob={opt.flip_prob})")
        print(f"   - Rotation: {opt.use_rotation} (prob={opt.rotation_prob}, degrees={opt.rotation_degrees})")
        print(f"   - TPS: {opt.use_tps} (prob={opt.tps_prob}, displacement={opt.tps_displacement})")

    train_dataset_videos = [
        VideosDataset(
            video_data_root=video_data_root,
            flow_data_root=video_data_root,
            mask_data_root=video_data_root,
            imagenet_folder=opt.data_root_imagenet,
            annotation_file_path=annotation_file_path,
            image_size=opt.image_size,
            image_transform=torch_transforms.Compose(transforms_video),
            real_reference_probability=opt.real_reference_probability,
            ref_augmentation=ref_augmentation,
        )
        for video_data_root, annotation_file_path in zip(opt.video_data_root_list,
                                                         opt.annotation_file_path_list)
    ]

    transforms_imagenet = [CenterPad_threshold(opt.image_size), RGB2Lab(), ToTensor(), Normalize()]
    extra_reference_transform = [
        torch_transforms.RandomHorizontalFlip(0.5),
        torch_transforms.RandomResizedCrop(480, (0.98, 1.0), ratio=(0.8, 1.2)),
    ]

    train_dataset_imagenet = VideosDataset_ImageNet(
        imagenet_data_root=opt.data_root_imagenet,
        pairs_file=opt.imagenet_pairs_file,
        image_size=opt.image_size,
        transforms_imagenet=transforms_imagenet,
        brightnessjitter=5,
        extra_reference_transform=extra_reference_transform,
        real_reference_probability=opt.real_reference_probability,
        ref_augmentation=ref_augmentation,
    )

    dataset_combined = ConcatDataset(train_dataset_videos + [train_dataset_imagenet])
    data_loader = prepare_dataloader_simple(dataset_combined,
                                           batch_size=opt.batch_size,
                                           pin_memory=False,
                                           num_workers=opt.workers)
    return data_loader

def save_checkpoints(saved_path, seed=None):
    os.makedirs(saved_path, exist_ok=True)

    torch.save(nonlocal_net.state_dict(), os.path.join(saved_path, "nonlocal_net.pth"))
    torch.save(colornet.state_dict(), os.path.join(saved_path, "colornet.pth"))
    torch.save(discriminator.state_dict(), os.path.join(saved_path, "discriminator.pth"))
    torch.save(embed_net.state_dict(), os.path.join(saved_path, "embed_net.pth"))

    learning_state = {
        "epoch": epoch_num,
        "total_iter": total_iter,
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
        "optimizer_schedule_g": step_optim_scheduler_g.state_dict(),
        "optimizer_schedule_d": step_optim_scheduler_d.state_dict(),
        "seed": seed,
    }

    torch.save(learning_state, os.path.join(saved_path, "learning_state.pth"))

def load_checkpoint(checkpoint_dir):
    """載入檢查點進行繼承訓練"""
    try:
        print(f"Loading checkpoint from {checkpoint_dir}")

        nonlocal_net.load_state_dict(load_params(os.path.join(checkpoint_dir, "nonlocal_net.pth"), device))
        colornet.load_state_dict(load_params(os.path.join(checkpoint_dir, "colornet.pth"), device))
        discriminator.load_state_dict(load_params(os.path.join(checkpoint_dir, "discriminator.pth"), device))
        embed_net_params = load_params(os.path.join(checkpoint_dir, "embed_net.pth"), device)
        embed_net.load_state_dict(embed_net_params)

        learning_state_path = os.path.join(checkpoint_dir, "learning_state.pth")
        if os.path.exists(learning_state_path):
            learning_checkpoint = torch.load(learning_state_path)
            optimizer_g.load_state_dict(learning_checkpoint["optimizer_g"])
            optimizer_d.load_state_dict(learning_checkpoint["optimizer_d"])

            start_epoch = learning_checkpoint['epoch'] + 1
            total_iter = learning_checkpoint['total_iter']

            saved_seed = learning_checkpoint.get("seed", None)
            if saved_seed is not None:
                print(f"✅ This checkpoint was trained with seed: {saved_seed}")
            else:
                print("⚠️  Warning: No seed information found in checkpoint")

            print(f"Successfully loaded checkpoint from epoch {learning_checkpoint['epoch']}")
            print(f"Resuming training from epoch {start_epoch}")
            print(f"Note: Learning rate scheduler will be recalculated for remaining epochs")

            return start_epoch, total_iter, saved_seed
        else:
            print("Warning: learning_state.pth not found. Starting optimizers from scratch.")
            return 1, 0, None

    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Starting training from scratch...")
        return 1, 0, None

def training_logger(step_count, avg_losses):
    if (step_count % opt.checkpoint_step == 0):
        print(f"\n=== Step {step_count} (Epoch {epoch_num}) ===")
        print(f"L1 Loss: {avg_losses['l1_loss']:.6f}")
        print(f"Perceptual Loss: {avg_losses['perceptual_loss']:.6f}")
        print(f"Contextual Loss: {avg_losses['contextual_loss']:.6f}")
        print(f"Smoothness Loss: {avg_losses['smoothness_loss']:.6f}")
        print(f"Generator Loss: {avg_losses['generator_loss']:.6f}")
        print(f"Discriminator Loss: {avg_losses['discriminator_loss']:.6f}")
        print(f"Total Loss: {avg_losses['total_loss']:.6f}")
        print("=" * 50)

        if opt.use_wandb:
            wandb.log({
                "train/total_loss": avg_losses['total_loss'],
                "train/l1_loss": avg_losses['l1_loss'],
                "train/perceptual_loss": avg_losses['perceptual_loss'],
                "train/contextual_loss": avg_losses['contextual_loss'],
                "train/smoothness_loss": avg_losses['smoothness_loss'],
                "train/discriminator_loss": avg_losses['discriminator_loss'],
                "train/generator_loss": avg_losses['generator_loss'],
                "train/opt_g_lr_1": step_optim_scheduler_g.get_last_lr()[0],
                "train/opt_g_lr_2": step_optim_scheduler_g.get_last_lr()[1],
                "train/opt_d_lr": step_optim_scheduler_d.get_last_lr()[0],
                "step": step_count
            })

def parse(parser, save=True):
    opt = parser.parse_args()
    args = vars(opt)

    print("------------------------------ Options -------------------------------")
    for k, v in sorted(args.items()):
        print("%s: %s" % (str(k), str(v)))
    print("-------------------------------- End ---------------------------------")

    if save:
        file_name = os.path.join("opt.txt")
        with open(file_name, "wt") as opt_file:
            opt_file.write(os.path.basename(sys.argv[0]) + " " + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + "\n")
            opt_file.write("------------------------------ Options -------------------------------\n")
            for k, v in sorted(args.items()):
                opt_file.write("%s: %s\n" % (str(k), str(v)))
            opt_file.write("-------------------------------- End ---------------------------------\n")
    return opt

def gpu_setup():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(0)
    return device

if __name__ == "__main__":
    opt = parse(parser)
    opt.video_data_root_list = opt.video_data_root_list.split(",")
    opt.annotation_file_path_list = opt.annotation_file_path_list.split(",")

    if opt.use_wandb:
        print("Initializing WandB...")
        if opt.wandb_token != "":
            try:
                wandb.login(key=opt.wandb_token)
            except:
                pass
        wandb.init(
            project="video-colorization",
            group=f"{opt.wandb_name} {datetime.now(tz=ZoneInfo('Asia/Ho_Chi_Minh')).strftime('%Y/%m/%d_%H-%M-%S')}"
        )

    device = gpu_setup()
    data_loader = load_data()
    print(f"Device: {device}")

    colornet = ColorVidNet(opt.ic).to(device)
    nonlocal_net = WarpNet().to(device)
    discriminator = Discriminator_x64_224(ndf=64).to(device)
    weighted_layer_color = WeightedAverage_color().to(device)
    nonlocal_weighted_layer = NonlocalWeightedAverage().to(device)
    embed_net = SwinModel(pretrained_model=opt.pretrained_model, device=device).to(device)

    print("-" * 59)
    print("|    TYPE   |          Model name            | Num params |")
    print("-" * 59)

    colornet_params = print_num_params(colornet)
    nonlocal_net_params = print_num_params(nonlocal_net)
    discriminator_params = print_num_params(discriminator)
    weighted_layer_color_params = print_num_params(weighted_layer_color)
    nonlocal_weighted_layer_params = print_num_params(nonlocal_weighted_layer)
    embed_net_params = print_num_params(embed_net)
    print("-" * 59)
    print(
        f"|   TOTAL   |                                | {('{:,}'.format(colornet_params+nonlocal_net_params+discriminator_params+weighted_layer_color_params+nonlocal_weighted_layer_params+embed_net_params)).rjust(10)} |"
    )
    print("-" * 59)

    if opt.use_wandb:
        wandb.watch(discriminator, log="all", log_freq=opt.checkpoint_step, idx=0)
        wandb.watch(embed_net, log="all", log_freq=opt.checkpoint_step, idx=1)
        wandb.watch(colornet, log="all", log_freq=opt.checkpoint_step, idx=2)
        wandb.watch(nonlocal_net, log="all", log_freq=opt.checkpoint_step, idx=3)

    perceptual_loss_fn = Perceptual_loss(opt.domain_invariant, opt.weight_perceptual)
    contextual_forward_loss = ContextualLoss_forward().to(device)

    optimizer_g = optim.AdamW(
        [
            {"params": nonlocal_net.parameters(), "lr": opt.lr},
            {"params": colornet.parameters(), "lr": 2 * opt.lr}
        ],
        betas=(0.5, 0.999),
        eps=1e-5,
        amsgrad=True,
    )

    optimizer_d = optim.AdamW(
        filter(lambda p: p.requires_grad, discriminator.parameters()),
        lr=opt.lr,
        betas=(0.5, 0.999),
        amsgrad=True,
    )

    total_steps = (len(data_loader) // opt.accumulation_steps) * opt.epoch
    step_optim_scheduler_g = PolynomialLR(
        optimizer_g,
        step_size=opt.lr_step,
        iter_warmup=0,
        iter_max=total_steps,
        power=0.9,
        min_lr=1e-8
    )
    step_optim_scheduler_d = PolynomialLR(
        optimizer_d,
        step_size=opt.lr_step,
        iter_warmup=0,
        iter_max=total_steps,
        power=0.9,
        min_lr=1e-8
    )

    downsampling_by2 = nn.AvgPool2d(kernel_size=2).to(device)
    loss_handler = LossHandler()

    start_epoch = 1
    total_iter = 0
    TRAINING_SEED = None

    if opt.resume and opt.resume_checkpoint_dir:
        start_epoch, total_iter, TRAINING_SEED = load_checkpoint(opt.resume_checkpoint_dir)

        remaining_epochs = opt.epoch - start_epoch + 1
        remaining_steps = (len(data_loader) // opt.accumulation_steps) * remaining_epochs

        print(f"Recalculating learning rate scheduler:")
        print(f"  Remaining epochs: {remaining_epochs}")
        print(f"  Remaining steps: {remaining_steps}")

        step_optim_scheduler_g = PolynomialLR(
            optimizer_g,
            step_size=opt.lr_step,
            iter_warmup=0,
            iter_max=remaining_steps,
            power=0.9,
            min_lr=1e-8
        )
        step_optim_scheduler_d = PolynomialLR(
            optimizer_d,
            step_size=opt.lr_step,
            iter_warmup=0,
            iter_max=remaining_steps,
            power=0.9,
            min_lr=1e-8
        )
    elif opt.load_pretrained_model:
        nonlocal_net.load_state_dict(load_params(os.path.join(opt.pretrained_model_dir, "nonlocal_net.pth"), device))
        colornet.load_state_dict(load_params(os.path.join(opt.pretrained_model_dir, "colornet.pth"), device))
        discriminator.load_state_dict(load_params(os.path.join(opt.pretrained_model_dir, "discriminator.pth"), device))
        embed_net_params = load_params(os.path.join(opt.pretrained_model_dir, "embed_net.pth"), device)
        embed_net.load_state_dict(embed_net_params)

        if os.path.exists(os.path.join(opt.pretrained_model_dir, "learning_state.pth")):
            learning_checkpoint = torch.load(os.path.join(opt.pretrained_model_dir, "learning_state.pth"))
            optimizer_g.load_state_dict(learning_checkpoint["optimizer_g"])
            optimizer_d.load_state_dict(learning_checkpoint["optimizer_d"])
            step_optim_scheduler_g.load_state_dict(learning_checkpoint["optimizer_schedule_g"])
            step_optim_scheduler_d.load_state_dict(learning_checkpoint["optimizer_schedule_d"])
            total_iter = learning_checkpoint['total_iter']
            start_epoch = learning_checkpoint['epoch']+1
            TRAINING_SEED = learning_checkpoint.get("seed", None)
        print(f"Loaded pretrained model from {opt.pretrained_model_dir}")
        if TRAINING_SEED is not None:
            print(f"✅ Pretrained model was trained with seed: {TRAINING_SEED}")
    else:
        TRAINING_SEED = torch.initial_seed()
        print(f"✅ Starting training from scratch with auto-generated seed: {TRAINING_SEED}")

    if opt.save_interval_steps > 0:
        print(f"✅ Checkpoint save strategy:")
        print(f"   - Every {opt.save_interval_steps} steps → checkpoints/epoch_{{epoch}}_latest/ (overwrite)")
        print(f"   - Every epoch end → checkpoints/epoch_{{epoch}}/ (permanent)")
    else:
        print(f"✅ Checkpoint save strategy:")
        print(f"   - Every epoch end → checkpoints/epoch_{{epoch}}/ (permanent)")

    print(f"✅ Loss averaging: interval average over {opt.checkpoint_step} steps")

    for epoch_num in range(start_epoch, opt.epoch+1):
        accumulated_losses = {
            'total_loss': 0.0,
            'l1_loss': 0.0,
            'perceptual_loss': 0.0,
            'contextual_loss': 0.0,
            'smoothness_loss': 0.0,
            'discriminator_loss': 0.0,
            'generator_loss': 0.0
        }
        step_count = 0
        accumulation_count = 0  # ✅ 新增：记录累加次数

        train_progress_bar = tqdm(
            data_loader,
            desc=f'Epoch {epoch_num}[Training]',
            position=0,
            leave=False
        )

        optimizer_g.zero_grad()

        for iter, sample in enumerate(train_progress_bar):
            total_iter += 1

            (I_current_lab, I_reference_lab,
             self_ref_flag, curr_frame_path, ref_path) = sample

            I_current_lab = I_current_lab.to(device)
            I_reference_lab = I_reference_lab.to(device)
            self_ref_flag = self_ref_flag.to(device)

            I_current_l = I_current_lab[:, 0:1, :, :]
            I_current_ab = I_current_lab[:, 1:3, :, :]
            I_reference_l = I_reference_lab[:, 0:1, :, :]
            I_reference_ab = I_reference_lab[:, 1:3, :, :]
            I_reference_rgb = tensor_lab2rgb(torch.cat((uncenter_l(I_reference_l), uncenter_ab(I_reference_ab)), dim=1))

            features_B = embed_net(I_reference_rgb)
            B_feat_0, B_feat_1, B_feat_2, B_feat_3 = features_B

            I_current_ab_predict, I_current_nonlocal_lab_predict = frame_colorization(
                IA_l=I_current_l,
                IB_lab=I_reference_lab,
                features_B=features_B,
                embed_net=embed_net,
                colornet=colornet,
                nonlocal_net=nonlocal_net,
                luminance_noise=opt.luminance_noise,
            )
            I_current_lab_predict = torch.cat((I_current_l, I_current_ab_predict), dim=1)

            discriminator_loss = torch.tensor(0.0, device=device)
            if opt.weight_gan > 0:
                fake_data_lab = torch.cat((
                    I_current_l, I_current_ab_predict
                ), dim=1)
                real_data_lab = torch.cat((
                    I_current_l, I_current_ab
                ), dim=1)

                if opt.permute_data:
                    batch_index = torch.arange(-1, opt.batch_size - 1, dtype=torch.long)
                    real_data_lab = real_data_lab[batch_index, ...]

                discriminator_loss = discriminator_loss_fn(real_data_lab, fake_data_lab, discriminator)
                discriminator_loss.backward()
                optimizer_d.step()
                optimizer_d.zero_grad()

            l1_loss = l1_loss_fn(I_current_ab, I_current_ab_predict) * opt.weight_l1

            generator_loss = torch.tensor(0.0, device=device)
            if epoch_num > opt.epoch_train_discriminator and opt.weight_gan > 0:
                generator_loss = generator_loss_fn(real_data_lab, fake_data_lab, discriminator, opt.weight_gan, device)

            I_predict_rgb = tensor_lab2rgb(torch.cat((uncenter_l(I_current_l), uncenter_ab(I_current_ab_predict)), dim=1))
            pred_feat_0, pred_feat_1, pred_feat_2, pred_feat_3 = embed_net(I_predict_rgb)

            I_current_rgb = tensor_lab2rgb(torch.cat((uncenter_l(I_current_l), uncenter_ab(I_current_ab)), dim=1))
            A_feat_0, _, _, A_feat_3 = embed_net(I_current_rgb)

            perceptual_loss = perceptual_loss_fn(A_feat_3, pred_feat_3)

            if abs(opt.weight_contextual) < 1e-8:
                contextual_loss_total = torch.tensor(0.0, device=device)
            else:
                contextual_style5_1 = torch.mean(contextual_forward_loss(pred_feat_3, B_feat_3.detach())) * 8
                contextual_style4_1 = torch.mean(contextual_forward_loss(pred_feat_2, B_feat_2.detach())) * 4
                contextual_style3_1 = torch.mean(contextual_forward_loss(pred_feat_1, B_feat_1.detach())) * 2
                contextual_style2_1 = torch.mean(contextual_forward_loss(pred_feat_0, B_feat_0.detach()))

                contextual_loss_total = (
                    contextual_style5_1 + contextual_style4_1 + contextual_style3_1 + contextual_style2_1
                ) * opt.weight_contextual

            smoothness_loss = smoothness_loss_fn(
                I_current_l,
                I_current_lab,
                I_current_ab_predict,
                A_feat_0,
                weighted_layer_color,
                nonlocal_weighted_layer,
                weight_smoothness=opt.weight_smoothness,
                weight_nonlocal_smoothness=opt.weight_nonlocal_smoothness,
                device=device
            )

            total_loss = (l1_loss + perceptual_loss + contextual_loss_total + smoothness_loss) / opt.accumulation_steps
            if epoch_num > opt.epoch_train_discriminator:
                total_loss += generator_loss / opt.accumulation_steps

            total_loss.backward()

            # ✅ 修改：直接累加，不除以actual_accumulation
            accumulated_losses['total_loss'] += (total_loss.item() * opt.accumulation_steps)
            accumulated_losses['l1_loss'] += l1_loss.item()
            accumulated_losses['perceptual_loss'] += perceptual_loss.item()
            accumulated_losses['contextual_loss'] += contextual_loss_total.item()
            accumulated_losses['smoothness_loss'] += smoothness_loss.item()
            accumulated_losses['discriminator_loss'] += discriminator_loss.item()
            if epoch_num > opt.epoch_train_discriminator:
                accumulated_losses['generator_loss'] += generator_loss.item()

            if (iter + 1) % opt.accumulation_steps == 0 or (iter + 1) == len(data_loader):
                step_count += 1
                accumulation_count += 1  # ✅ 新增：累加计数

                optimizer_g.step()
                optimizer_g.zero_grad()

                step_optim_scheduler_g.step()
                step_optim_scheduler_d.step()

                # ✅ 修改：在checkpoint时计算平均、打印、重置
                if step_count % opt.checkpoint_step == 0:
                    avg_losses = {key: value / accumulation_count for key, value in accumulated_losses.items()}
                    training_logger(step_count, avg_losses)

                    # 重置累加器
                    for key in accumulated_losses:
                        accumulated_losses[key] = 0.0
                    accumulation_count = 0

                # 按步数保存checkpoint（覆盖策略）
                if opt.save_interval_steps > 0 and step_count % opt.save_interval_steps == 0:
                    latest_checkpoint_path = os.path.join(opt.checkpoint_dir, f"epoch_{epoch_num}_latest")
                    save_checkpoints(latest_checkpoint_path, seed=TRAINING_SEED)
                    print(f"✅ Saved latest checkpoint to {latest_checkpoint_path} (will be overwritten)")

        # Epoch结束时永久保存
        save_checkpoints(os.path.join(opt.checkpoint_dir, f"epoch_{epoch_num}"), seed=TRAINING_SEED)
        print(f"✅ Saved permanent checkpoint: epoch_{epoch_num}")

    if opt.use_wandb:
        wandb.finish()
