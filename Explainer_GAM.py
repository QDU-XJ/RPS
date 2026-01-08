import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import os
import torch.nn.functional as F   # 放在文件最前面
from model2 import MultiModalClassifier          # ← 你的网络
from dataset4 import MIMDataset                 # ← 你的数据集
import  cv2
# ---------------- 1. 读 1 张真实样本 ----------------
INPUT_DIR = r"F:\RSHS2\localH_crop"
exl_path = r'F:\RSHS\Omics\Lasso\n_sum.csv'
label_excel = r'F:\RSHS2\label.xlsx'

# 2）包装模型，仅接受 image 作为输入
class Wrapper(torch.nn.Module):
    def __init__(self, model, fixed_omics):
        super().__init__()
        self.model = model
        self.fixed_omics = fixed_omics
    def forward(self, x):
        return self.model(x, self.fixed_omics.to(x.device))

train_ds = MIMDataset(INPUT_DIR, exl_path, label_excel)
print(len(train_ds))
sample = train_ds[0]
flag=0
for sample in train_ds:
    img_np   = sample['image']          # (3,126,126)  float32 0~1
    omics_np = sample['omics']          # (48,)
    label    = sample['label']
    case_id  = sample['case_id']
    maska = sample['mask_A']
    case_id = sample['case_id']

    img   = torch.from_numpy(img_np).unsqueeze(0)   # [1,3,126,126]
    omics = omics_np.unsqueeze(0).float()           # [1,48]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MultiModalClassifier().to(device)
    model.eval()

    wrapped = Wrapper(model, omics)

    # 3）目标层
    target_layer = model.resnet_adjust[0]      # Conv2d(512→12)

    # 4）GradCAM（**新版接口**）
    cam = GradCAM(model=wrapped, target_layers=[target_layer])

    grayscale_cam = cam(input_tensor=img.to(device), targets=None)[0]  # (126,126)
    # print(grayscale_cam.shape)

    if grayscale_cam.shape != (256, 256):
        grayscale_cam = torch.from_numpy(grayscale_cam).unsqueeze(0).unsqueeze(0)
        grayscale_cam = F.interpolate(grayscale_cam,
                                      size=(256, 256),
                                      mode='bilinear',
                                      align_corners=False)
        grayscale_cam = grayscale_cam.squeeze().numpy()

    grayscale_cam = np.clip(grayscale_cam, 0, 1)

    # 5）可视化
    img_vis = img_np[:,:,0] #
    print(img_vis.shape)
    print(grayscale_cam.shape)

    # 1）灰度 CAM → 伪彩色
    heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)   # (126,126,3)

    # 2）与原图叠加
    img_vis = img_np[:,:,0] #                # (126,126,3) 原图
    img_vis_3ch = np.stack([img_vis] * 3, axis=-1)
    alpha = 0.5                                         # 透明度
    cam_img = (alpha * heatmap / 255.0) + (1 - alpha) * img_vis_3ch
    cam_img = np.clip(cam_img, 0, 1)

    print(cam_img.shape)
    maska = np.broadcast_to(maska, (3,256, 256))
    maska = np.transpose(maska, (1, 2, 0))
    print(np.max(maska))
    cam_img = cam_img * maska
    # 3）显示或保存
    plt.figure(figsize=(5,5))
    im=plt.imshow(cam_img)
    plt.axis('off')
    plt.colorbar(im)
    plt.title(f'Grad-CAM')
    save_path = f'CAM\gradcam_rgb_{case_id}.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    # plt.show()
    print(f'已保存：{os.path.abspath(save_path)}')