from src.models.CNN.ColorVidNet import ColorVidNet
from src.models.vit.embed import SwinModel
from src.models.CNN.NonlocalNet import WarpNet
from src.models.CNN.FrameColor import frame_colorization
import torch
from src.models.vit.utils import load_params
import os
import cv2
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

def uncenter_ab(ab):
    return ab * 127.0

class SwinTExCo:
    def __init__(self, weights_path, swin_backbone='swinv2-cr-t-224', device=None):
        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.embed_net = SwinModel(pretrained_model=swin_backbone, device=self.device).to(self.device)
        self.nonlocal_net = WarpNet(feature_channel=128).to(self.device)
        self.colornet = ColorVidNet(4).to(self.device)  # ✅ 修改1: 從 7 改為 4
        
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
        
    def __preprocess_reference(self, img):
        color_enhancer = IE.Color(img)
        img = color_enhancer.enhance(1.5)
        return img
    
    def __upscale_image(self, large_IA_l, I_current_ab_predict):
        H, W = large_IA_l.shape[2:]
        large_current_ab_predict = torch.nn.functional.interpolate(I_current_ab_predict, 
                                                                size=(H,W), 
                                                                mode="bilinear", 
                                                                align_corners=False)
        large_IA_lab = torch.cat((large_IA_l, uncenter_ab(large_current_ab_predict)), dim=1)
        large_current_rgb_predict = tensor_lab2rgb(large_IA_lab)
        return large_current_rgb_predict.cpu()
    
    def __proccess_sample(self, curr_frame, I_reference_lab, features_B):
        """
        處理單幀圖像 - 單幀上色模式
        Args:
            curr_frame: 當前幀 (PIL Image)
            I_reference_lab: 參考圖的LAB表示
            features_B: 參考圖的特徵
        """
        large_IA_lab = ToTensor()(RGB2Lab()(curr_frame)).unsqueeze(0)
        large_IA_l = large_IA_lab[:, 0:1, :, :].to(self.device)
        
        IA_lab = self.processor(curr_frame)
        IA_lab = IA_lab.unsqueeze(0).to(self.device)
        IA_l = IA_lab[:, 0:1, :, :]
        
        # ✅ 修改2: 移除 placeholder 生成

        with torch.no_grad():
            # ✅ 修改3: frame_colorization 調用移除 placeholder 參數
            I_current_ab_predict, _ = frame_colorization(
                IA_l,
                I_reference_lab,
                features_B,
                self.embed_net,
                self.nonlocal_net,
                self.colornet,
                luminance_noise=0,
                temperature=1e-10,
                joint_training=False
            )
            
        IA_predict_rgb = self.__upscale_image(large_IA_l, I_current_ab_predict)
        IA_predict_rgb = (IA_predict_rgb.squeeze(0).cpu().numpy() * 255.)
        IA_predict_rgb = np.clip(IA_predict_rgb, 0, 255).astype(np.uint8)
        
        return IA_predict_rgb
    
    def predict_video(self, video, ref_image):
        """
        視頻上色 - 單幀模式 (每幀獨立處理)
        """
        ref_image = self.__preprocess_reference(ref_image)
        
        IB_lab = self.processor(ref_image)
        IB_lab = IB_lab.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            I_reference_lab = IB_lab
            I_reference_l = I_reference_lab[:, 0:1, :, :]
            I_reference_ab = I_reference_lab[:, 1:3, :, :]
            I_reference_rgb = tensor_lab2rgb(torch.cat((uncenter_l(I_reference_l), uncenter_ab(I_reference_ab)), dim=1)).to(self.device)
            features_B = self.embed_net(I_reference_rgb)
        
        while video.isOpened():
            ret, curr_frame = video.read()
            
            if not ret:
                break
            
            curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)
            curr_frame = Image.fromarray(curr_frame)
            
            # ✅ 每幀獨立處理,不依賴前一幀
            IA_predict_rgb = self.__proccess_sample(curr_frame, I_reference_lab, features_B)
            
            IA_predict_rgb = IA_predict_rgb.transpose(1,2,0)
            
            yield IA_predict_rgb

        video.release()

    def predict_image(self, image, ref_image):
        """
        單張圖像上色
        """
        ref_image = self.__preprocess_reference(ref_image)
        
        IB_lab = self.processor(ref_image)
        IB_lab = IB_lab.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            I_reference_lab = IB_lab
            I_reference_l = I_reference_lab[:, 0:1, :, :]
            I_reference_ab = I_reference_lab[:, 1:3, :, :]
            I_reference_rgb = tensor_lab2rgb(torch.cat((uncenter_l(I_reference_l), uncenter_ab(I_reference_ab)), dim=1)).to(self.device)
            features_B = self.embed_net(I_reference_rgb)
        
        curr_frame = image
        IA_predict_rgb = self.__proccess_sample(curr_frame, I_reference_lab, features_B)
        
        IA_predict_rgb = IA_predict_rgb.transpose(1,2,0)
        
        return IA_predict_rgb
    
if __name__ == "__main__":
    model = SwinTExCo('checkpoints/epoch_2/')
    
    # 單張圖像上色
    target_image = Image.open('../dataset/NTIRE/ntire-dataset/ntire_videos/001/frame_000027.jpg').convert('RGB')
    ref_image = Image.open('../dataset/NTIRE/ntire-dataset/ntire_videos/001/frame_000030.jpg').convert('RGB')
    
    colorized = model.predict_image(target_image, ref_image)
    result = Image.fromarray(colorized)
    result.save('sample_output/epoch2.jpg')
    
    print("單張圖像上色完成！")
    
    # 視頻上色範例
    # video = cv2.VideoCapture('sample_input/video_2.mp4')
    # fps = video.get(cv2.CAP_PROP_FPS)
    # width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # video_writer = cv2.VideoWriter('sample_output/video_2_ref_2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    # 	
    # for colorized_frame in model.predict_video(video, ref_image):
    #     video_writer.write(colorized_frame)
    # 
    # video_writer.release()



