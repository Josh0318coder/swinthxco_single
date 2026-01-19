"""

Reference Image Augmentation for Fusion Training

Provides geometric augmentation strategies to prevent shortcut learning
when using video first frame as reference image.

Strategies:
    - RandomResizedCrop: Random scale and crop variation (NEW)
    - TPSAugmentation: Thin Plate Spline non-rigid transformation (primary)
    - HorizontalFlip: Global mirroring
    - Rotation: Slight rotation (±10 degrees)

All augmentations preserve color accuracy while breaking spatial alignment,
forcing the model to learn semantic matching rather than pixel copying.
"""

import random
import numpy as np
from PIL import Image
import cv2


class RandomResizedCropAugmentation:
    """
    Random Resized Crop augmentation: scale and crop variation

    Randomly crops a region from the image and resizes it back to original size.
    This simulates different scales and viewpoints, improving scale invariance.

    Args:
        scale: Crop area range as ratio of original image (default: (0.8, 1.0))
        ratio: Aspect ratio range of crop (default: (0.9, 1.1))
        augmentation_prob: Probability of applying crop (default: 0.5)
    """

    def __init__(self,
                 scale=(0.8, 1.0),
                 ratio=(0.9, 1.1),
                 augmentation_prob=0.5):
        self.scale = scale
        self.ratio = ratio
        self.augmentation_prob = augmentation_prob

    def __call__(self, reference_pil):
        """
        Apply random resized crop to reference image

        Args:
            reference_pil: PIL.Image (RGB)

        Returns:
            cropped_pil: PIL.Image (RGB) with same size as input
        """
        if random.random() > self.augmentation_prob:
            return reference_pil

        try:
            W, H = reference_pil.size

            # Random crop area (as ratio of original image)
            area = W * H
            target_area = random.uniform(self.scale[0], self.scale[1]) * area

            # Random aspect ratio
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])

            # Calculate crop dimensions
            crop_w = int(round(np.sqrt(target_area * aspect_ratio)))
            crop_h = int(round(np.sqrt(target_area / aspect_ratio)))

            # Ensure crop fits within image
            if crop_w > W:
                crop_w = W
                crop_h = int(crop_w / aspect_ratio)
            if crop_h > H:
                crop_h = H
                crop_w = int(crop_h * aspect_ratio)

            # Random crop position
            if crop_w < W:
                left = random.randint(0, W - crop_w)
            else:
                left = 0

            if crop_h < H:
                top = random.randint(0, H - crop_h)
            else:
                top = 0

            # Crop and resize back to original size
            cropped = reference_pil.crop((left, top, left + crop_w, top + crop_h))
            resized = cropped.resize((W, H), Image.LANCZOS)

            return resized

        except Exception as e:
            print(f"Warning: RandomResizedCrop augmentation failed: {e}")
            return reference_pil


class TPSAugmentation:
    """
    Thin Plate Spline augmentation: local non-rigid deformation

    Preserves colors but breaks spatial alignment at fine-grained level,
    forcing the model to learn dense semantic correspondence.

    Args:
        num_control_points: Grid size for control points (default: 5 → 5×5=25 points)
        displacement_range: Max displacement as ratio of image size (default: 0.1 = ±10%)
        augmentation_prob: Probability of applying TPS (default: 0.7)
    """

    def __init__(self,
                 num_control_points=5,
                 displacement_range=0.1,
                 augmentation_prob=0.7):
        self.num_control_points = num_control_points
        self.displacement_range = displacement_range
        self.augmentation_prob = augmentation_prob

    def __call__(self, reference_pil):
        """
        Apply TPS-like transformation to reference image using grid warping

        Args:
            reference_pil: PIL.Image (RGB)

        Returns:
            warped_pil: PIL.Image (RGB) with spatial warping
        """
        if random.random() > self.augmentation_prob:
            return reference_pil

        try:
            img = np.array(reference_pil)
            H, W = img.shape[:2]

            # Generate control points grid
            grid_size = self.num_control_points
            x = np.linspace(0, W-1, grid_size)
            y = np.linspace(0, H-1, grid_size)

            # Create displacement grid
            grid_x, grid_y = np.meshgrid(
                np.linspace(0, W-1, grid_size),
                np.linspace(0, H-1, grid_size)
            )

            # Add random displacement to control points
            displaced_x = grid_x + np.random.uniform(
                -self.displacement_range * W,
                self.displacement_range * W,
                grid_x.shape
            )
            displaced_y = grid_y + np.random.uniform(
                -self.displacement_range * H,
                self.displacement_range * H,
                grid_y.shape
            )

            # Clamp to image boundaries
            displaced_x = np.clip(displaced_x, 0, W-1)
            displaced_y = np.clip(displaced_y, 0, H-1)

            # Create dense pixel grid for interpolation
            map_x, map_y = np.meshgrid(np.arange(W), np.arange(H))
            map_x = map_x.astype(np.float32)
            map_y = map_y.astype(np.float32)

            # Interpolate displacement field from control points to all pixels
            # Using linear interpolation for smooth warping
            from scipy.interpolate import griddata

            # Flatten control point grids
            points = np.column_stack([grid_x.flatten(), grid_y.flatten()])
            values_x = displaced_x.flatten()
            values_y = displaced_y.flatten()

            # Interpolate to full resolution
            grid_points = np.column_stack([map_x.flatten(), map_y.flatten()])
            map_x = griddata(points, values_x, grid_points, method='cubic', fill_value=0).reshape(H, W).astype(np.float32)
            map_y = griddata(points, values_y, grid_points, method='cubic', fill_value=0).reshape(H, W).astype(np.float32)

            # Apply warping using remap
            warped = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

            return Image.fromarray(warped)

        except Exception as e:
            # Fallback: return original image if warping fails
            print(f"Warning: TPS augmentation failed: {e}")
            return reference_pil


class RotationAugmentation:
    """
    Rotation augmentation: slight global rotation

    Simulates camera angle changes, breaking spatial alignment while
    preserving semantic structure.

    Args:
        degrees: Max rotation angle in degrees (default: 10 → ±10°)
        augmentation_prob: Probability of applying rotation (default: 0.3)

        fill_mode: Border filling mode (default: 'reflect')
    """

    def __init__(self,
                 degrees=10,
                 augmentation_prob=0.3,
                 fill_mode='reflect'):
        self.degrees = degrees
        self.augmentation_prob = augmentation_prob
        self.fill_mode = fill_mode

    def __call__(self, reference_pil):
        """
        Apply rotation to reference image

        Args:
            reference_pil: PIL.Image (RGB)

        Returns:
            rotated_pil: PIL.Image (RGB)
        """
        if random.random() > self.augmentation_prob:
            return reference_pil

        try:
            # Random angle
            angle = random.uniform(-self.degrees, self.degrees)

            img = np.array(reference_pil)
            H, W = img.shape[:2]

            # Rotation matrix
            center = (W / 2, H / 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)

            # Border mode mapping
            border_modes = {
                'reflect': cv2.BORDER_REFLECT,
                'replicate': cv2.BORDER_REPLICATE,
                'constant': cv2.BORDER_CONSTANT
            }
            border_mode = border_modes.get(self.fill_mode, cv2.BORDER_REFLECT)

            # Apply rotation
            rotated = cv2.warpAffine(
                img, M, (W, H),
                borderMode=border_mode
            )

            return Image.fromarray(rotated)

        except Exception as e:
            print(f"Warning: Rotation augmentation failed: {e}")
            return reference_pil


class HorizontalFlipAugmentation:
    """
    Horizontal flip augmentation: global mirroring

    Completely breaks left-right spatial correspondence, forcing
    the model to learn semantic matching rather than position-based copying.

    Args:
        augmentation_prob: Probability of flipping (default: 0.5)
    """

    def __init__(self, augmentation_prob=0.5):
        self.augmentation_prob = augmentation_prob

    def __call__(self, reference_pil):
        """
        Apply horizontal flip to reference image

        Args:
            reference_pil: PIL.Image (RGB)

        Returns:
            flipped_pil: PIL.Image (RGB)
        """
        if random.random() > self.augmentation_prob:
            return reference_pil

        return reference_pil.transpose(Image.FLIP_LEFT_RIGHT)


class GeometricAugmentation:
    """
    Combined geometric augmentation pipeline

    Applies transformations in order: RandomResizedCrop → Flip → Rotation → TPS
    This order ensures:
    1. Scale variation first (RandomResizedCrop)
    2. Global transformations next (flip, rotation)
    3. Local transformations last (TPS)
    4. Each transformation's details are preserved

    Args:
        use_crop: Enable random resized crop (default: False)
        use_flip: Enable horizontal flip (default: True)
        use_rotation: Enable rotation (default: True)
        use_tps: Enable TPS (default: True)
        crop_prob: Crop probability (default: 0.5)
        crop_scale: Crop scale range (default: (0.8, 1.0))
        crop_ratio: Crop aspect ratio range (default: (0.9, 1.1))
        flip_prob: Flip probability (default: 0.5)
        rotation_prob: Rotation probability (default: 0.3)
        rotation_degrees: Max rotation angle (default: 10)
        tps_prob: TPS probability (default: 0.7)
        tps_displacement: TPS displacement range (default: 0.1)
        tps_control_points: TPS grid size (default: 5)
    """

    def __init__(self,
                 use_crop=False,
                 use_flip=True,
                 use_rotation=True,
                 use_tps=True,
                 crop_prob=0.5,
                 crop_scale=(0.8, 1.0),
                 crop_ratio=(0.9, 1.1),
                 flip_prob=0.5,
                 rotation_prob=0.3,
                 rotation_degrees=10,
                 tps_prob=0.7,
                 tps_displacement=0.1,
                 tps_control_points=5):

        self.use_crop = use_crop
        self.use_flip = use_flip
        self.use_rotation = use_rotation
        self.use_tps = use_tps

        # Initialize augmentations
        if use_crop:
            self.crop = RandomResizedCropAugmentation(
                scale=crop_scale,
                ratio=crop_ratio,
                augmentation_prob=crop_prob
            )

        if use_flip:
            self.flip = HorizontalFlipAugmentation(augmentation_prob=flip_prob)

        if use_rotation:
            self.rotation = RotationAugmentation(
                degrees=rotation_degrees,
                augmentation_prob=rotation_prob,
                fill_mode='reflect'
            )

        if use_tps:
            self.tps = TPSAugmentation(
                num_control_points=tps_control_points,
                displacement_range=tps_displacement,
                augmentation_prob=tps_prob
            )

    def __call__(self, reference_pil):
        """
        Apply geometric augmentation pipeline

        Order: RandomResizedCrop → Flip → Rotation → TPS (coarse to fine)

        Args:
            reference_pil: PIL.Image (RGB)

        Returns:
            augmented_pil: PIL.Image (RGB)
        """
        # Step 0: Random resized crop (scale variation)
        if self.use_crop:
            reference_pil = self.crop(reference_pil)

        # Step 1: Horizontal flip (global)
        if self.use_flip:
            reference_pil = self.flip(reference_pil)

        # Step 2: Rotation (global)
        if self.use_rotation:
            reference_pil = self.rotation(reference_pil)

        # Step 3: TPS (local)
        if self.use_tps:
            reference_pil = self.tps(reference_pil)

        return reference_pil
