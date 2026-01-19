"""
可视化TPS数据增强效果
"""
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os

# 导入增强类
from augmentation import GeometricAugmentation

def visualize_augmentation(image_path, output_dir='aug_visualization', num_samples=8):
    """
    可视化单张图像的多次增强效果

    Args:
        image_path: 输入图像路径
        output_dir: 输出目录
        num_samples: 生成的增强样本数量
    """
    os.makedirs(output_dir, exist_ok=True)

    # 读取图像
    image = Image.open(image_path).convert('RGB')

    # 初始化增强器（与训练参数一致）
    augmentor = GeometricAugmentation(
        use_flip=True,
        use_rotation=True,
        use_tps=True,
        flip_prob=0.5,
        rotation_prob=0.3,
        rotation_degrees=10,
        tps_prob=0.7,
        tps_displacement=0.1,
        tps_control_points=5
    )

    # 创建网格图
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f'TPS Data Augmentation Visualization\nImage: {os.path.basename(image_path)}',
                 fontsize=16)

    # 第一张显示原图
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    # 生成增强样本
    for idx in range(1, num_samples + 1):
        # ✅ 修复：直接传入PIL Image，augmentor会返回PIL Image
        augmented_pil = augmentor(image)

        row = idx // 3
        col = idx % 3

        axes[row, col].imshow(augmented_pil)
        axes[row, col].set_title(f'Augmented #{idx}', fontsize=12)
        axes[row, col].axis('off')

    plt.tight_layout()

    # 保存结果
    output_path = os.path.join(output_dir, f'aug_{os.path.basename(image_path)}')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 保存可视化结果: {output_path}")

    plt.close()


def visualize_augmentation_comparison(image_path, output_dir='aug_visualization'):
    """
    对比显示：原图 vs 仅Flip vs 仅Rotation vs 仅TPS vs 全部增强
    """
    os.makedirs(output_dir, exist_ok=True)

    image = Image.open(image_path).convert('RGB')

    # 不同的增强配置
    configs = [
        ("Original", None),
        ("Flip Only", GeometricAugmentation(use_flip=True, use_rotation=False, use_tps=False, flip_prob=1.0)),
        ("Rotation Only", GeometricAugmentation(use_flip=False, use_rotation=True, use_tps=False, rotation_prob=1.0, rotation_degrees=10)),
        ("TPS Only", GeometricAugmentation(use_flip=False, use_rotation=False, use_tps=True, tps_prob=1.0, tps_displacement=0.1)),
        ("All Combined", GeometricAugmentation(use_flip=True, use_rotation=True, use_tps=True,
                                               flip_prob=1.0, rotation_prob=1.0, tps_prob=1.0,
                                               rotation_degrees=10, tps_displacement=0.1))
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Augmentation Type Comparison', fontsize=16)

    for idx, (title, augmentor) in enumerate(configs):
        row = idx // 3
        col = idx % 3

        if augmentor is None:
            result = image
        else:
            # ✅ 修复：直接传入PIL Image
            result = augmentor(image)

        axes[row, col].imshow(result)
        axes[row, col].set_title(title, fontsize=12, fontweight='bold')
        axes[row, col].axis('off')

    # 隐藏最后一个子图
    axes[1, 2].axis('off')

    plt.tight_layout()

    output_path = os.path.join(output_dir, f'comparison_{os.path.basename(image_path)}')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 保存对比结果: {output_path}")

    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='可视化TPS数据增强效果')
    parser.add_argument('--image', type=str, required=True, help='输入图像路径')
    parser.add_argument('--output_dir', type=str, default='aug_visualization', help='输出目录')
    parser.add_argument('--mode', type=str, choices=['grid', 'comparison', 'both'],
                       default='both', help='可视化模式')
    parser.add_argument('--num_samples', type=int, default=8, help='网格模式的样本数量')

    args = parser.parse_args()

    if args.mode in ['grid', 'both']:
        print("生成网格可视化...")
        visualize_augmentation(args.image, args.output_dir, args.num_samples)

    if args.mode in ['comparison', 'both']:
        print("生成对比可视化...")
        visualize_augmentation_comparison(args.image, args.output_dir)

    print(f"\n✅ 完成！结果保存在: {args.output_dir}")
