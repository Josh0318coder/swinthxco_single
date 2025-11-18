import sys
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from src.utils import uncenter_l

# ===============================
# 直接3136×3136計算方法
# ===============================

def direct_similarity_calculation(theta, phi):
    """
    直接計算3136×3136相似度矩陣
    
    輸入:
    - theta: [batch_size, 128, 3136] - A特徵
    - phi: [batch_size, 128, 3136] - B特徵
    
    輸出:
    - f: [batch_size, 3136, 3136] - 相似度矩陣
    """
    # 正規化theta
    theta = theta - theta.mean(dim=-1, keepdim=True) 
    theta_norm = torch.norm(theta, 2, 1, keepdim=True) + sys.float_info.epsilon
    theta = torch.div(theta, theta_norm)
    theta_permute = theta.permute(0, 2, 1)  # [batch_size, 3136, 128]

    # 正規化phi
    phi = phi - phi.mean(dim=-1, keepdim=True) 
    phi_norm = torch.norm(phi, 2, 1, keepdim=True) + sys.float_info.epsilon
    phi = torch.div(phi, phi_norm)
    
    # 計算相似度矩陣
    f = torch.matmul(theta_permute, phi)  # [batch_size, 3136, 3136]
    
    return f

# ===============================
# max similarity mapping
# ===============================

def mapping_to_gray(B_lab, f):
    """
    
    輸入:
    - B_lab: [batch_size, 3, 224, 224] - reference image
    - f: [batch_size, 3136, 3136] - 相似度矩陣
    
    輸出:
    - y: [batch_size, 3, 224, 224] - mapped color
    - similarty map: [batch_size, 1, 224, 224] -  similarty map
    """
    f_similarity = f.unsqueeze_(dim=1)
    similarity_map, similarity_index = torch.max(f_similarity, -1, keepdim=True)
    similarity_map = similarity_map.view(B_lab.shape[0], 1, 56, 56)

    # f can be negative
    f_WTA = f / 0.005
    #f_div_C = F.softmax(f_WTA.squeeze_(), dim=-1)  # 2*1936*1936;

    # downsample the reference color
    B_lab = F.avg_pool2d(B_lab, 4)
    B_lab = B_lab.view(B_lab.shape[0], 3 , -1)
    B_lab = B_lab.permute(0, 2, 1)  # 2*1936*channel

    # multiply the corr map with color
    max_indices = torch.argmax(f_WTA.squeeze_(), dim = -1)
    y = B_lab[torch.arange(B_lab.shape[0]).unsqueeze(1), max_indices]
    #y = torch.matmul(f_div_C, B_lab)  # 2*1936*channel
    y = y.permute(0, 2, 1).contiguous()
    y = y.view(B_lab.shape[0], 3, 56, 56)  # 2*3*44*44
    y =F.interpolate(y, scale_factor=4)
    similarity_map = F.interpolate(similarity_map, scale_factor=4)

    return y, similarity_map

def find_local_patch(x, patch_size):
    """
    > We take a tensor `x` and return a tensor `x_unfold` that contains all the patches of size
    `patch_size` in `x`

    Args:
      x: the input tensor
      patch_size: the size of the patch to be extracted.
    """

    N, C, H, W = x.shape
    x_unfold = F.unfold(x, kernel_size=(patch_size, patch_size), padding=(patch_size // 2, patch_size // 2), stride=(1, 1))

    return x_unfold.view(N, x_unfold.shape[1], H, W)


class WeightedAverage(nn.Module):
    def __init__(
        self,
    ):
        super(WeightedAverage, self).__init__()

    def forward(self, x_lab, patch_size=3, alpha=1, scale_factor=1):
        """
        It takes a 3-channel image (L, A, B) and returns a 2-channel image (A, B) where each pixel is a
        weighted average of the A and B values of the pixels in a 3x3 neighborhood around it

        Args:
          x_lab: the input image in LAB color space
          patch_size: the size of the patch to use for the local average. Defaults to 3
          alpha: the higher the alpha, the smoother the output. Defaults to 1
          scale_factor: the scale factor of the input image. Defaults to 1

        Returns:
          The output of the forward function is a tensor of size (batch_size, 2, height, width)
        """
        # alpha=0: less smooth; alpha=inf: smoother
        x_lab = F.interpolate(x_lab, scale_factor=scale_factor)
        l = x_lab[:, 0:1, :, :]
        a = x_lab[:, 1:2, :, :]
        b = x_lab[:, 2:3, :, :]
        local_l = find_local_patch(l, patch_size)
        local_a = find_local_patch(a, patch_size)
        local_b = find_local_patch(b, patch_size)
        local_difference_l = (local_l - l) ** 2
        correlation = nn.functional.softmax(-1 * local_difference_l / alpha, dim=1)

        return torch.cat(
            (
                torch.sum(correlation * local_a, dim=1, keepdim=True),
                torch.sum(correlation * local_b, dim=1, keepdim=True),
            ),
            1,
        )


class WeightedAverage_color(nn.Module):
    """
    smooth the image according to the color distance in the LAB space
    """

    def __init__(
        self,
    ):
        super(WeightedAverage_color, self).__init__()

    def forward(self, x_lab, x_lab_predict, patch_size=3, alpha=1, scale_factor=1):
        """
        It takes the predicted a and b channels, and the original a and b channels, and finds the
        weighted average of the predicted a and b channels based on the similarity of the original a and
        b channels to the predicted a and b channels

        Args:
          x_lab: the input image in LAB color space
          x_lab_predict: the predicted LAB image
          patch_size: the size of the patch to use for the local color correction. Defaults to 3
          alpha: controls the smoothness of the output. Defaults to 1
          scale_factor: the scale factor of the input image. Defaults to 1

        Returns:
          The return is the weighted average of the local a and b channels.
        """
        """ alpha=0: less smooth; alpha=inf: smoother """
        x_lab = F.interpolate(x_lab, scale_factor=scale_factor)
        l = uncenter_l(x_lab[:, 0:1, :, :])
        a = x_lab[:, 1:2, :, :]
        b = x_lab[:, 2:3, :, :]
        a_predict = x_lab_predict[:, 1:2, :, :]
        b_predict = x_lab_predict[:, 2:3, :, :]
        local_l = find_local_patch(l, patch_size)
        local_a = find_local_patch(a, patch_size)
        local_b = find_local_patch(b, patch_size)
        local_a_predict = find_local_patch(a_predict, patch_size)
        local_b_predict = find_local_patch(b_predict, patch_size)

        local_color_difference = (local_l - l) ** 2 + (local_a - a) ** 2 + (local_b - b) ** 2
        # so that sum of weights equal to 1
        correlation = nn.functional.softmax(-1 * local_color_difference / alpha, dim=1)

        return torch.cat(
            (
                torch.sum(correlation * local_a_predict, dim=1, keepdim=True),
                torch.sum(correlation * local_b_predict, dim=1, keepdim=True),
            ),
            1,
        )


class NonlocalWeightedAverage(nn.Module):
    def __init__(
        self,
    ):
        super(NonlocalWeightedAverage, self).__init__()

    def forward(self, x_lab, feature, patch_size=3, alpha=0.1, scale_factor=1):
        """
        It takes in a feature map and a label map, and returns a smoothed label map

        Args:
            x_lab: the input image in LAB color space
            feature: the feature map of the input image
            patch_size: the size of the patch to be used for the correlation matrix. Defaults to 3
            alpha: the higher the alpha, the smoother the output.
            scale_factor: the scale factor of the input image. Defaults to 1

        Returns:
            weighted_ab is the weighted ab channel of the image.
        """
        # alpha=0: less smooth; alpha=inf: smoother
        # input feature is normalized feature
        x_lab = F.interpolate(x_lab, scale_factor=scale_factor)
        batch_size, channel, height, width = x_lab.shape
        feature = F.interpolate(feature, size=(height, width))
        batch_size = x_lab.shape[0]
        x_ab = x_lab[:, 1:3, :, :].view(batch_size, 2, -1)
        x_ab = x_ab.permute(0, 2, 1)

        local_feature = find_local_patch(feature, patch_size)
        local_feature = local_feature.view(batch_size, local_feature.shape[1], -1)

        correlation_matrix = torch.matmul(local_feature.permute(0, 2, 1), local_feature)
        correlation_matrix = nn.functional.softmax(correlation_matrix / alpha, dim=-1)

        weighted_ab = torch.matmul(correlation_matrix, x_ab)
        weighted_ab = weighted_ab.permute(0, 2, 1).contiguous()
        weighted_ab = weighted_ab.view(batch_size, 2, height, width)
        return weighted_ab


class CorrelationLayer(nn.Module):
    def __init__(self, search_range):
        super(CorrelationLayer, self).__init__()
        self.search_range = search_range

    def forward(self, x1, x2, alpha=1, raw_output=False, metric="similarity"):
        """
        It takes two tensors, x1 and x2, and returns a tensor of shape (batch_size, (search_range * 2 +
        1) ** 2, height, width) where each element is the dot product of the corresponding patch in x1
        and x2

        Args:
          x1: the first image
          x2: the image to be warped
          alpha: the temperature parameter for the softmax function. Defaults to 1
          raw_output: if True, return the raw output of the network, otherwise return the softmax
        output. Defaults to False
          metric: "similarity" or "subtraction". Defaults to similarity

        Returns:
          The output of the forward function is a softmax of the correlation volume.
        """
        shape = list(x1.size())
        shape[1] = (self.search_range * 2 + 1) ** 2
        cv = torch.zeros(shape).to(torch.device("cuda"))

        for i in range(-self.search_range, self.search_range + 1):
            for j in range(-self.search_range, self.search_range + 1):
                if i < 0:
                    slice_h, slice_h_r = slice(None, i), slice(-i, None)
                elif i > 0:
                    slice_h, slice_h_r = slice(i, None), slice(None, -i)
                else:
                    slice_h, slice_h_r = slice(None), slice(None)

                if j < 0:
                    slice_w, slice_w_r = slice(None, j), slice(-j, None)
                elif j > 0:
                    slice_w, slice_w_r = slice(j, None), slice(None, -j)
                else:
                    slice_w, slice_w_r = slice(None), slice(None)

                if metric == "similarity":
                    cv[:, (self.search_range * 2 + 1) * i + j, slice_h, slice_w] = (
                        x1[:, :, slice_h, slice_w] * x2[:, :, slice_h_r, slice_w_r]
                    ).sum(1)
                else:  # patchwise subtraction
                    cv[:, (self.search_range * 2 + 1) * i + j, slice_h, slice_w] = -(
                        (x1[:, :, slice_h, slice_w] - x2[:, :, slice_h_r, slice_w_r]) ** 2
                    ).sum(1)

        # TODO sigmoid?
        if raw_output:
            return cv
        else:
            return nn.functional.softmax(cv / alpha, dim=1)


class WTA_scale(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, scale=1e-4):
        """
        In the forward pass we receive a Tensor containing the input and return a
        Tensor containing the output. You can cache arbitrary Tensors for use in the
        backward pass using the save_for_backward method.
        """
        activation_max, index_max = torch.max(input, -1, keepdim=True)
        input_scale = input * scale  # default: 1e-4
        # input_scale = input * scale  # default: 1e-4
        output_max_scale = torch.where(input == activation_max, input, input_scale)

        mask = (input == activation_max).type(torch.float)
        ctx.save_for_backward(input, mask)
        return output_max_scale

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, mask = ctx.saved_tensors
        mask_ones = torch.ones_like(mask)
        mask_small_ones = torch.ones_like(mask) * 1e-4
        # mask_small_ones = torch.ones_like(mask) * 1e-4

        grad_scale = torch.where(mask == 1, mask_ones, mask_small_ones)
        grad_input = grad_output.clone() * grad_scale
        return grad_input, None


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(ResidualBlock, self).__init__()
        self.padding1 = nn.ReflectionPad2d(padding)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0, stride=stride)
        self.bn1 = nn.InstanceNorm2d(out_channels)
        self.prelu = nn.PReLU()
        self.padding2 = nn.ReflectionPad2d(padding)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0, stride=stride)
        self.bn2 = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.padding1(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.padding2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.prelu(out)
        return out


class WarpNet(nn.Module):
    """input is Al, Bl, channel = 1, range~[0,255]"""

    def __init__(self, feature_channel=128):
        super(WarpNet, self).__init__()
        self.feature_channel = feature_channel
        self.in_channels = self.feature_channel * 4
        self.inter_channels = 256
        # 44*44
        self.layer2_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            # nn.Conv2d(128, 128, kernel_size=3, padding=0, stride=1),
            # nn.Conv2d(96, 128, kernel_size=3, padding=20, stride=1),
            nn.Conv2d(96, 128, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(128),
            nn.PReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, self.feature_channel, kernel_size=3, padding=0, stride=2),
            nn.InstanceNorm2d(self.feature_channel),
            nn.PReLU(),
            nn.Dropout(0.2),
        )
        self.layer3_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            # nn.Conv2d(256, 128, kernel_size=3, padding=0, stride=1),
            # nn.Conv2d(192, 128, kernel_size=3, padding=10, stride=1),
            nn.Conv2d(192, 128, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(128),
            nn.PReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, self.feature_channel, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(self.feature_channel),
            nn.PReLU(),
            nn.Dropout(0.2),
        )

        # 22*22->44*44
        self.layer4_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            # nn.Conv2d(512, 256, kernel_size=3, padding=0, stride=1),
            # nn.Conv2d(384, 256, kernel_size=3, padding=5, stride=1),
            nn.Conv2d(384, 256, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(256),
            nn.PReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, self.feature_channel, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(self.feature_channel),
            nn.PReLU(),
            #nn.Upsample(scale_factor=2),
            #nn.Dropout(0.2),
        )

        # 11*11->44*44
        self.layer5_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            # nn.Conv2d(1024, 256, kernel_size=3, padding=0, stride=1),
            # nn.Conv2d(768, 256, kernel_size=2, padding=2, stride=1),
            nn.Conv2d(768, 256, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(256),
            nn.PReLU(),
            nn.Upsample(scale_factor=2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, self.feature_channel, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(self.feature_channel),
            nn.PReLU(),
            #nn.Upsample(scale_factor=2),
            #nn.Dropout(0.2),
        )

        self.layer = nn.Sequential(
            ResidualBlock(self.feature_channel * 4, self.feature_channel * 4, kernel_size=3, padding=1, stride=1),
            ResidualBlock(self.feature_channel * 4, self.feature_channel * 4, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(self.feature_channel * 4, self.feature_channel * 2, kernel_size=3, padding=0, stride=1),
            nn.ReflectionPad2d(1),
            nn.Conv2d(self.feature_channel * 2, self.feature_channel * 1, kernel_size=3, padding=0, stride=1),
            nn.ReflectionPad2d(1),
        )
	
        self.theta = nn.Conv2d(
            in_channels=self.feature_channel, out_channels=self.feature_channel, kernel_size=1, stride=1, padding=0
        )
        self.phi = nn.Conv2d(
            in_channels=self.feature_channel, out_channels=self.feature_channel, kernel_size=1, stride=1, padding=0)

        self.upsampling = nn.Upsample(scale_factor=4)

    def forward(
        self,
        B_lab_map,
        A_relu2_1,
        A_relu3_1,
        A_relu4_1,
        A_relu5_1,
        B_relu2_1,
        B_relu3_1,
        B_relu4_1,
        B_relu5_1,
        temperature=0.001 * 5,
        detach_flag=False,
        indices_a = None
    ):
        batch_size = B_lab_map.shape[0]
        channel = B_lab_map.shape[1]
        image_height = B_lab_map.shape[2]
        image_width = B_lab_map.shape[3]
        feature_height = int(image_height / 4)
        feature_width = int(image_width / 4)
        device = B_lab_map.device
        
        if indices_a is None:
            indices_a = torch.arange(49, device=device).view(1, 1, 49, 1).expand(batch_size, 1, 49, 1)

        # scale feature size to 44*44
        A_feature2_1 = self.layer2_1(A_relu2_1)
        #print(f"feature2_1 shape:{A_relu2_1.shape}{A_feature2_1.shape}")
        B_feature2_1 = self.layer2_1(B_relu2_1)
        A_feature3_1 = self.layer3_1(A_relu3_1)
        #print(f"feature3_1 shape:{A_relu3_1.shape}{A_feature3_1.shape}")
        B_feature3_1 = self.layer3_1(B_relu3_1)
        A_feature4_1 = self.layer4_1(A_relu4_1)
        #print(f"feature4_1 shape:{A_relu4_1.shape}{A_feature4_1.shape}")
        B_feature4_1 = self.layer4_1(B_relu4_1)
        A_feature5_1 = self.layer5_1(A_relu5_1)
        #print(f"feature5_1 shape:{A_relu5_1.shape}{A_feature5_1.shape}")
        B_feature5_1 = self.layer5_1(B_relu5_1)
        
# ============ Down sample for multiscale ============   
        #A_feature3_1_down = F.interpolate(A_feature3_1, scale_factor=0.5, mode='bilinear', align_corners=False)
        #B_feature3_1_down = F.interpolate(B_feature3_1, scale_factor=0.5, mode='bilinear', align_corners=False)
        #A_feature4_1_down = F.interpolate(A_feature4_1, scale_factor=0.5, mode='bilinear', align_corners=False)
        #B_feature4_1_down = F.interpolate(B_feature4_1, scale_factor=0.5, mode='bilinear', align_corners=False)
        #A_feature5_1_down = F.interpolate(A_feature5_1, scale_factor=0.25, mode='bilinear', align_corners=False)
        #B_feature5_1_down = F.interpolate(B_feature5_1, scale_factor=0.25, mode='bilinear', align_corners=False)
        #print(f"feature_down shape:{A_feature3_1_down.shape}{A_feature4_1_down.shape}{A_feature5_1_down.shape}")
        
# ============ Up sample for concatenate ============       
        A_feature4_1_up = F.interpolate(A_feature4_1, scale_factor=2, mode='bilinear', align_corners=False)
        B_feature4_1_up = F.interpolate(B_feature4_1, scale_factor=2, mode='bilinear', align_corners=False)
        A_feature5_1_up = F.interpolate(A_feature5_1, scale_factor=2, mode='bilinear', align_corners=False)
        B_feature5_1_up = F.interpolate(B_feature5_1, scale_factor=2, mode='bilinear', align_corners=False)
        #print(f"feature_up shape:{A_feature4_1_up.shape}{A_feature5_1_up.shape}")

        # concatenate features
        if A_feature5_1_up.shape[2] != A_feature2_1.shape[2] or A_feature5_1_up.shape[3] != A_feature2_1.shape[3]:
            A_feature5_1 = F.pad(A_feature5_1_up, (0, 0, 1, 1), "replicate")
            B_feature5_1 = F.pad(B_feature5_1_up, (0, 0, 1, 1), "replicate")

        #print(f"concate shape:{torch.cat((A_feature2_1, A_feature3_1, A_feature4_1_up, A_feature5_1_up), 1).shape}")
        A_features = self.layer(torch.cat((A_feature2_1, A_feature3_1, A_feature4_1_up, A_feature5_1_up), 1))
        B_features = self.layer(torch.cat((B_feature2_1, B_feature3_1, B_feature4_1_up, B_feature5_1_up), 1))
        #print(f"feature_concate shape:{A_features.shape}{B_features.shape}")

        #print(f"Multiscale shape:\n{A_feature5_1_down.shape}\n{A_feature4_1_down.shape}\n{A_feature3_1_down.shape}\n{self.theta(A_features).shape}")
        
        # pairwise cosine similarity
        theta = self.theta(A_features).view(batch_size, self.feature_channel, -1)  # 2*256*(feature_height*feature_width)
        phi = self.phi(B_features).view(batch_size, self.feature_channel, -1)  # 2*256*(feature_height*feature_width)
        #print(f"theta shape:{theta.shape}")
        '''theta = theta - theta.mean(dim=-1, keepdim=True)  # center the feature
        theta_norm = torch.norm(theta, 2, 1, keepdim=True) + sys.float_info.epsilon
        theta = torch.div(theta, theta_norm)
        theta_permute = theta.permute(0, 2, 1)  # 2*(feature_height*feature_width)*256
        phi = phi - phi.mean(dim=-1, keepdim=True)  # center the feature
        phi_norm = torch.norm(phi, 2, 1, keepdim=True) + sys.float_info.epsilon
        phi = torch.div(phi, phi_norm)
        f = torch.matmul(theta_permute, phi)  # 2*(feature_height*feature_width)*(feature_height*feature_width)'''

# ========================== test ========================
        '''A_feature5_1_down = A_feature5_1_down.view(batch_size, self.feature_channel, -1)
        B_feature5_1_down = B_feature5_1_down.view(batch_size, self.feature_channel, -1)
        A_feature_7_7 = A_feature5_1_down - A_feature5_1_down.mean(dim=-1, keepdim=True)  # center the feature
        A_feature_7_7_norm = torch.norm(A_feature_7_7, 2, 1, keepdim=True) + sys.float_info.epsilon
        A_feature_7_7 = torch.div(A_feature_7_7, A_feature_7_7_norm)
        A_feature_7_7_permute = A_feature_7_7.permute(0, 2, 1)  # 2*(feature_height*feature_width)*256
        
        B_feature7_7 = B_feature5_1_down - B_feature5_1_down.mean(dim=-1, keepdim=True)  # center the feature
        B_feature7_7_norm = torch.norm(B_feature7_7, 2, 1, keepdim=True) + sys.float_info.epsilon
        B_feature7_7 = torch.div(B_feature7_7, B_feature7_7_norm)
        f = torch.matmul(A_feature_7_7_permute, B_feature7_7)  # 2*(feature_height*feature_width)*(feature_height*feature_width)
        if detach_flag:
            f = f.detach()'''
# =======================================================

        torch.cuda.synchronize()
        start = time.time()
        memory_before = torch.cuda.memory_allocated() / 1024**2
        f = direct_similarity_calculation(theta, phi)
        #f = direct_similarity_calculation(A_feature5_1_down, B_feature5_1_down)
        if detach_flag:
            f = f.detach()
        '''indices_b = torch.max(direct_similarity_calculation(A_feature5_1_down,B_feature5_1_down), -1, keepdim =True)[-1]
        #print(f"indices_b shape:{indices_b.shape}")
        indices_a_out, indices_b_out = fast_batch_mapping(indices_a, indices_b, 8)
        #print(f"indices_a_out shape:{indices_a_out.shape}")
        max_similarities, similarities = optimized_vectorized_region_similarity(theta, phi, indices_a_out, indices_b_out)
        #print(f"max_similarities shape:{max_similarities.shape}")
        reordered_similarities = optimized_reorder_similarities(similarities, indices_a_out)
        #print(f"reordered_similarities shape:{reordered_similarities.shape}")
        y, similarity_map = mapping_to_gray(B_lab_map, reordered_similarities)'''
	
            
        y, similarity_map = mapping_to_gray(B_lab_map, f)
        torch.cuda.synchronize()
        memory_after = torch.cuda.memory_allocated() / 1024**2         
        print(f"time:{(time.time()-start)*1000:.2f}ms, memory used:{memory_after-memory_before:.1f}MB")
        
        '''f_similarity = f.unsqueeze_(dim=1)
        similarity_map, similarity_index = torch.max(f_similarity, -1, keepdim=True)
        print(f"similarity shape:{similarity_map.shape}{similarity_index.shape}")
        similarity_map = similarity_map.view(batch_size, 1, feature_height, feature_width)

        # f can be negative
        f_WTA = f if WTA_scale_weight == 1 else WTA_scale.apply(f, WTA_scale_weight)
        f_WTA = f_WTA / temperature
        f_div_C = F.softmax(f_WTA.squeeze_(), dim=-1)  # 2*1936*1936;
        print(f"f_div_C shape:{f_div_C.shape}")

        # downsample the reference color
        B_lab = F.avg_pool2d(B_lab_map, 4)
        B_lab = B_lab.view(batch_size, channel, -1)
        B_lab = B_lab.permute(0, 2, 1)  # 2*1936*channel

        # multiply the corr map with color
        y = torch.matmul(f_div_C, B_lab)  # 2*1936*channel
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, channel, feature_height, feature_width)  # 2*3*44*44
        y = self.upsampling(y)
        similarity_map = self.upsampling(similarity_map)'''

        return y, similarity_map
