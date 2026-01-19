"""
SwinTExCo Inference Script with Intermediate Visualization

Features:
- Single image colorization
- Batch dataset processing
- Optional visualization of nonlocal fusion (similarity × ref_ab) at 56×56
- Similarity map heatmap visualization (upsampled or raw 56×56)

Usage:
    # Single image
    python inference_swintexco_with_vis.py \
        --mode single \
        --weights checkpoints/epoch_40 \
        --target_image gray.jpg \
        --ref_image color_ref.jpg \
        --output result.jpg

    # Dataset with all visualizations
    python inference_swintexco_with_vis.py \
        --mode dataset \
        --weights checkpoints/epoch_40 \
        --input_root /path/to/dataset \
        --output_root /path/to/output \
        --save_fusion \
        --save_similarity \
        --save_raw_similarity \
        --heatmap_colormap turbo
"""

from src.models.CNN.ColorVidNet import ColorVidNet
from src.models.vit.embed import SwinModel
from src.models.CNN.NonlocalNet import WarpNet
from src.models.CNN.FrameColor import frame_colorization
import torch
from src.models.vit.utils import load_params
import os
from PIL import Image
from PIL import ImageEnhance as IE
import torchvision.transforms as T
from src.utils import (
    RGB2Lab,
    ToTensor,
    Normalize,
    uncenter_l,
    tensor_lab2rgb
)
import numpy as np
import glob
import argparse
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def uncenter_ab(ab):
    return ab * 127.0


def rgb_to_grayscale(image):
    """Convert RGB image to grayscale (keeping 3 channels)"""
    gray = image.convert('L')
    return gray.convert('RGB')


def visualize_heatmap(tensor, colormap='turbo'):
    """
    Convert similarity map to colored heatmap

    Args:
        tensor: [1, 1, H, W] or [H, W] tensor
        colormap: 'turbo', 'viridis', 'jet', 'hot'

    Returns:
        PIL Image (RGB)
    """
    # Extract data
    if tensor.ndim == 4:
        data = tensor.squeeze().cpu().numpy()
    elif tensor.ndim == 3:
        data = tensor.squeeze(0).cpu().numpy()
    else:
        data = tensor.cpu().numpy()

    # Normalize to [0, 1]
    data_min, data_max = data.min(), data.max()
    if data_max - data_min > 1e-6:
        data = (data - data_min) / (data_max - data_min)
    else:
        data = np.zeros_like(data)

    # Apply colormap
    cmap = plt.get_cmap(colormap)
    colored = cmap(data)[:, :, :3]  # RGB only
    rgb_uint8 = (colored * 255).astype(np.uint8)

    return Image.fromarray(rgb_uint8)


class SwinTExCoWithVisualization:
    def __init__(self, weights_path, swin_backbone='swinv2-cr-t-224', device=None):
        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.embed_net = SwinModel(pretrained_model=swin_backbone, device=self.device).to(self.device)
        self.nonlocal_net = WarpNet(feature_channel=128).to(self.device)
        self.colornet = ColorVidNet(4).to(self.device)

        self.embed_net.eval()
        self.nonlocal_net.eval()
        self.colornet.eval()

        self.__load_models(self.embed_net, os.path.join(weights_path, "embed_net.pth"))
        self.__load_models(self.nonlocal_net, os.path.join(weights_path, "nonlocal_net.pth"))
        self.__load_models(self.colornet, os.path.join(weights_path, "colornet.pth"))

        self.processor = T.Compose([
            T.Resize((224,224)),
            RGB2Lab(),
            ToTensor(),
            Normalize()
        ])

    def __load_models(self, model, weight_path):
        params = load_params(weight_path, self.device)
        model.load_state_dict(params, strict=True)

    def __preprocess_reference(self, img, enhance_factor=1.5):
        """
        Preprocess reference image

        Args:
            img: PIL Image
            enhance_factor: Color saturation enhancement (1.0=no change, 1.5=+50%)
        """
        if enhance_factor != 1.0:
            color_enhancer = IE.Color(img)
            img = color_enhancer.enhance(enhance_factor)
        return img

    def __upscale_image(self, large_IA_l, I_current_ab_predict):
        """Upscale predicted ab to original size"""
        H, W = large_IA_l.shape[2:]
        large_current_ab_predict = torch.nn.functional.interpolate(
            I_current_ab_predict,
            size=(H, W),
            mode="bilinear",
            align_corners=False
        )
        large_IA_lab = torch.cat((large_IA_l, uncenter_ab(large_current_ab_predict)), dim=1)
        large_current_rgb_predict = tensor_lab2rgb(large_IA_lab)
        return large_current_rgb_predict.cpu()

    def __process_sample(self, curr_frame, I_reference_lab, features_B, return_intermediates=False):
        """
        Process single frame

        Args:
            curr_frame: PIL Image
            I_reference_lab: Reference LAB tensor
            features_B: Pre-computed reference features
            return_intermediates: If True, return fusion ab and similarity map

        Returns:
            If return_intermediates=False: (colorized_rgb,)
            If return_intermediates=True: (colorized_rgb, fusion_ab, similarity_map)
        """
        # Get original size
        large_IA_lab = ToTensor()(RGB2Lab()(curr_frame)).unsqueeze(0)
        large_IA_l = large_IA_lab[:, 0:1, :, :].to(self.device)

        # Resize to 224x224 for model
        IA_lab = self.processor(curr_frame)
        IA_lab = IA_lab.unsqueeze(0).to(self.device)
        IA_l = IA_lab[:, 0:1, :, :]

        with torch.no_grad():
            # Frame colorization
            if return_intermediates:
                # 獲取原始融合輸出（尺寸=輸入/4，無插值損失）
                colorization_results = frame_colorization(
                    IA_l,
                    I_reference_lab,
                    features_B,
                    self.embed_net,
                    self.nonlocal_net,
                    self.colornet,
                    luminance_noise=0,
                    temperature=1e-10,
                    joint_training=False,
                    return_raw=True
                )
                I_current_ab_predict, I_current_nonlocal_lab_predict, _, nonlocal_BA_lab_raw, similarity_map_raw = colorization_results
            else:
                # 默認模式：只獲取最終結果
                I_current_ab_predict, I_current_nonlocal_lab_predict = frame_colorization(
                    IA_l,
                    I_reference_lab,
                    features_B,
                    self.embed_net,
                    self.nonlocal_net,
                    self.colornet,
                    luminance_noise=0,
                    temperature=1e-10,
                    joint_training=False,
                    return_raw=False
                )

        # Upscale final result
        IA_predict_rgb = self.__upscale_image(large_IA_l, I_current_ab_predict)
        IA_predict_rgb = (IA_predict_rgb.squeeze(0).cpu().numpy() * 255.)
        IA_predict_rgb = np.clip(IA_predict_rgb, 0, 255).astype(np.uint8)
        IA_predict_rgb = IA_predict_rgb.transpose(1, 2, 0)

        if not return_intermediates:
            return (IA_predict_rgb,)

        # 使用原始融合輸出（無需下採樣，直接從NonlocalNet獲取）
        # 尺寸 = image_size / 4，對於224×224輸入 → 56×56，對於176×176輸入 → 44×44
        fusion_raw_ab = nonlocal_BA_lab_raw[:, 1:3, :, :]  # [1, 2, H/4, W/4] (原始融合ab)
        raw_h, raw_w = fusion_raw_ab.shape[2], fusion_raw_ab.shape[3]

        # 獲取對應尺寸的L通道
        IA_l_raw = torch.nn.functional.interpolate(
            IA_l,
            size=(raw_h, raw_w),
            mode="bilinear",
            align_corners=False
        )  # [1, 1, H/4, W/4]

        # 拼接為原始尺寸的LAB並轉換為RGB
        fusion_raw_rgb = tensor_lab2rgb(torch.cat((uncenter_l(IA_l_raw), uncenter_ab(fusion_raw_ab)), dim=1)).cpu()
        fusion_raw_rgb_np = (fusion_raw_rgb.squeeze(0).numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        # fusion_raw_rgb_np 形狀: [H/4, W/4, 3] (無插值損失的原始融合結果)

        # 保存原始相似度圖（56×56）
        raw_similarity_np = similarity_map_raw.squeeze().cpu().numpy()  # [H/4, W/4]

        # Upscale similarity map to original size
        H, W = large_IA_l.shape[2:]
        large_similarity = torch.nn.functional.interpolate(
            similarity_map_raw,
            size=(H, W),
            mode="bilinear",
            align_corners=False
        )

        return (IA_predict_rgb, fusion_raw_rgb_np, large_similarity, raw_similarity_np)

    def predict_image(self, image, ref_image):
        """Single image colorization (basic mode)"""
        ref_image = self.__preprocess_reference(ref_image)

        IB_lab = self.processor(ref_image)
        IB_lab = IB_lab.unsqueeze(0).to(self.device)

        with torch.no_grad():
            I_reference_lab = IB_lab
            I_reference_l = I_reference_lab[:, 0:1, :, :]
            I_reference_ab = I_reference_lab[:, 1:3, :, :]
            I_reference_rgb = tensor_lab2rgb(torch.cat((uncenter_l(I_reference_l), I_reference_ab), dim=1)).to(self.device)
            features_B = self.embed_net(I_reference_rgb)

        results = self.__process_sample(image, I_reference_lab, features_B, return_intermediates=False)
        return results[0]

    def predict_dataset(self, input_root, output_root,
                       image_extensions=['*.jpg', '*.png', '*.jpeg'],
                       save_fusion=False,
                       save_similarity=False,
                       save_raw_similarity=False,
                       heatmap_colormap='turbo',
                       enhance_factor=1.5):
        """
        Batch dataset processing with optional visualization

        Args:
            input_root: Input dataset root
            output_root: Output dataset root
            image_extensions: Supported image formats
            save_fusion: Save nonlocal fusion ab visualization
            save_similarity: Save similarity map heatmap (upsampled to original size)
            save_raw_similarity: Save raw similarity map (56×56, no upsampling)
            heatmap_colormap: Colormap for similarity ('turbo', 'viridis', 'jet', 'hot')
            enhance_factor: Reference image color enhancement (1.0=no change)
        """
        # Get all scene folders
        scene_folders = sorted([d for d in os.listdir(input_root)
                               if os.path.isdir(os.path.join(input_root, d))])

        print(f"Found {len(scene_folders)} scene folders")

        # Process each scene
        for scene_name in tqdm(scene_folders, desc="Processing scenes"):
            scene_input_path = os.path.join(input_root, scene_name)
            scene_output_path = os.path.join(output_root, scene_name)

            # Create output directories
            os.makedirs(scene_output_path, exist_ok=True)

            if save_fusion:
                fusion_output_path = os.path.join(output_root, f"{scene_name}_fusion")
                os.makedirs(fusion_output_path, exist_ok=True)

            if save_similarity:
                similarity_output_path = os.path.join(output_root, f"{scene_name}_similarity")
                os.makedirs(similarity_output_path, exist_ok=True)

            if save_raw_similarity:
                raw_similarity_output_path = os.path.join(output_root, f"{scene_name}_similarity_raw")
                os.makedirs(raw_similarity_output_path, exist_ok=True)

            # Get all image files
            image_files = []
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(scene_input_path, ext)))
                image_files.extend(glob.glob(os.path.join(scene_input_path, ext.upper())))
            image_files = sorted(image_files)

            if len(image_files) == 0:
                print(f"Warning: No images found in scene {scene_name}, skipping")
                continue

            print(f"\nProcessing scene '{scene_name}': {len(image_files)} images")

            # First frame as reference
            ref_image_path = image_files[0]
            ref_image = Image.open(ref_image_path).convert('RGB')
            print(f"  Reference image: {os.path.basename(ref_image_path)}")

            # Preprocess reference and extract features ONCE
            ref_image_processed = self.__preprocess_reference(ref_image, enhance_factor)
            IB_lab = self.processor(ref_image_processed)
            IB_lab = IB_lab.unsqueeze(0).to(self.device)

            with torch.no_grad():
                I_reference_lab = IB_lab
                I_reference_l = I_reference_lab[:, 0:1, :, :]
                I_reference_ab = I_reference_lab[:, 1:3, :, :]
                I_reference_rgb = tensor_lab2rgb(torch.cat((uncenter_l(I_reference_l), I_reference_ab), dim=1)).to(self.device)
                features_B = self.embed_net(I_reference_rgb)

            # Process each frame
            need_intermediates = save_fusion or save_similarity or save_raw_similarity

            for img_path in tqdm(image_files, desc=f"  {scene_name}", leave=False):
                # Read and convert to grayscale
                image = Image.open(img_path).convert('RGB')
                gray_image = rgb_to_grayscale(image)

                # Process with optional intermediates
                results = self.__process_sample(
                    gray_image,
                    I_reference_lab,
                    features_B,
                    return_intermediates=need_intermediates
                )

                # Unpack results
                if need_intermediates:
                    colorized, fusion_rgb, similarity, raw_similarity = results
                else:
                    colorized = results[0]

                # Save final result
                filename = os.path.basename(img_path)
                output_path = os.path.join(scene_output_path, filename)
                Image.fromarray(colorized).save(output_path)

                # Save fusion visualization
                if save_fusion:
                    fusion_path = os.path.join(fusion_output_path, filename)
                    Image.fromarray(fusion_rgb).save(fusion_path)

                # Save similarity heatmap (upsampled)
                if save_similarity:
                    sim_heatmap = visualize_heatmap(similarity, colormap=heatmap_colormap)
                    sim_path = os.path.join(similarity_output_path, filename)
                    sim_heatmap.save(sim_path)

                # Save raw similarity heatmap (56×56, no upsampling)
                if save_raw_similarity:
                    raw_sim_heatmap = visualize_heatmap(torch.from_numpy(raw_similarity), colormap=heatmap_colormap)
                    raw_sim_path = os.path.join(raw_similarity_output_path, filename)
                    raw_sim_heatmap.save(raw_sim_path)

            print(f"  ✅ Completed scene '{scene_name}'")

        print(f"\n✅ All done! Results saved to: {output_root}")
        if save_fusion:
            print(f"   Fusion visualization (56×56): {output_root}/*_fusion/")
        if save_similarity:
            print(f"   Similarity heatmaps (upsampled): {output_root}/*_similarity/")
        if save_raw_similarity:
            print(f"   Raw similarity heatmaps (56×56): {output_root}/*_similarity_raw/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SwinTExCo Inference with Visualization')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['single', 'dataset'],
                       help='Processing mode')
    parser.add_argument('--weights', type=str, required=True,
                       help='Model weights path')

    # Single image mode
    parser.add_argument('--target_image', type=str,
                       help='Target image path (required for single mode)')
    parser.add_argument('--ref_image', type=str,
                       help='Reference image path (required for single mode)')
    parser.add_argument('--output', type=str,
                       help='Output path (required for single mode)')

    # Dataset mode
    parser.add_argument('--input_root', type=str,
                       help='Input dataset root (required for dataset mode)')
    parser.add_argument('--output_root', type=str,
                       help='Output dataset root (required for dataset mode)')

    # Visualization options
    parser.add_argument('--save_fusion', action='store_true',
                       help='Save nonlocal fusion (similarity × ref_ab) visualization (56×56)')
    parser.add_argument('--save_similarity', action='store_true',
                       help='Save similarity map heatmap (upsampled to original size)')
    parser.add_argument('--save_raw_similarity', action='store_true',
                       help='Save raw similarity map heatmap (56×56, no upsampling)')
    parser.add_argument('--heatmap_colormap', type=str, default='turbo',
                       choices=['turbo', 'viridis', 'jet', 'hot'],
                       help='Colormap for similarity heatmap (default: turbo)')

    # Reference enhancement
    parser.add_argument('--enhance_factor', type=float, default=1.5,
                       help='Reference color enhancement factor (1.0=no change, default: 1.5)')

    args = parser.parse_args()

    # Initialize model
    print(f"Loading model weights: {args.weights}")
    model = SwinTExCoWithVisualization(args.weights)

    if args.mode == 'single':
        if not all([args.target_image, args.ref_image, args.output]):
            parser.error("--mode single requires --target_image, --ref_image, --output")

        print(f"Loading target image: {args.target_image}")
        print(f"Loading reference image: {args.ref_image}")

        target_image = Image.open(args.target_image).convert('RGB')
        ref_image = Image.open(args.ref_image).convert('RGB')

        colorized = model.predict_image(target_image, ref_image)
        result = Image.fromarray(colorized)
        result.save(args.output)

        print(f"✅ Single image colorization complete! Result saved to: {args.output}")

    elif args.mode == 'dataset':
        if not all([args.input_root, args.output_root]):
            parser.error("--mode dataset requires --input_root, --output_root")

        print(f"Input dataset: {args.input_root}")
        print(f"Output directory: {args.output_root}")
        if args.save_fusion:
            print("  Saving nonlocal fusion visualization (56×56)")
        if args.save_similarity:
            print(f"  Saving similarity heatmaps - upsampled (colormap: {args.heatmap_colormap})")
        if args.save_raw_similarity:
            print(f"  Saving raw similarity heatmaps - 56×56 (colormap: {args.heatmap_colormap})")
        print(f"  Reference enhancement: {args.enhance_factor}x")
        print("\nStarting batch processing...\n")

        model.predict_dataset(
            args.input_root,
            args.output_root,
            save_fusion=args.save_fusion,
            save_similarity=args.save_similarity,
            save_raw_similarity=args.save_raw_similarity,
            heatmap_colormap=args.heatmap_colormap,
            enhance_factor=args.enhance_factor
        )
