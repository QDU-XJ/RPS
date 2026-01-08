import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import shap
from model2 import MultiModalClassifier, WrapperModel
from dataset4 import MIMDataset
from torch.utils.data import DataLoader
import pandas as pd
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
base_model = MultiModalClassifier().to(device)
sizec=256
# 2. 加载 checkpoint，再提取模型权重
ckpt_path = 'model_checkpoints/best_model.pth'            # 改成你的真实路径
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)  # 或 True 若已确认文件安全

# 如果保存的是完整 checkpoint
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint  # 若当时只保存了 state_dict

base_model.load_state_dict(state_dict)

# 3. 设为评估模式（重要！）
base_model.eval()

# --------------------------------------------------
# 4. 包装成 WrapperModel，供后续 SHAP 使用
wrap_model = WrapperModel(base_model, img_size=(sizec, sizec)).to(device).eval()

# ---------------- 准备背景样本 -----------------
#随机背景 背景样本越多越稳定，但显存消耗线性增加
# N_bg = 2000
# bg_img  = torch.randn(N_bg, 3, 146, 146).to(device)
# bg_omics = torch.randn(N_bg, 48).to(device)

# A. 用真实数据切片做背景 / 测试
# 把训练集里 20 张真实图像 + 对应组学 作为 bg，再挑 1 张真实图像做 test_x，数值范围立刻拉开。

if __name__ == '__main__':
    INPUT_DIR = r"F:\RSHS2\localH_crop"
    exl_path = r'F:\RSHS\Omics\Lasso\n_sum.csv'
    label_excel = r'F:\RSHS2\label.xlsx'
    train_data = MIMDataset(INPUT_DIR, exl_path, label_excel)
    ## 测试数据集
    # 1. 从训练集 DataLoader 采样 20 条做背景
    bg_imgs, bg_omics = [], []
    set_seed(1)
    batch_size = 1
    bg_size = 110
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=0)
    for sample in train_loader:
        if len(bg_imgs) >= bg_size:
            break
        img = sample['image']  # [1, 126, 126, 3]
        img = img.permute(0, 3, 1, 2)  # [1, 3, 126, 126]
        bg_imgs.append(img)  # [1, 3, 146, 146]
        bg_omics.append(sample['omics'])

    # 拼 batch
    bg_imgs = torch.cat(bg_imgs, dim=0).to(device)  # (20, 3, 146, 146)
    bg_omics = torch.cat(bg_omics, dim=0).to(device)  # (20, 48)
    bg = torch.cat([bg_imgs.reshape(bg_size, -1), bg_omics], dim=1)

    # ---------------- 待解释样本 -------------------
    # 这里解释第 0 个测试样本
    # test_img   = torch.randn(1, 3, 146, 146).to(device)
    # test_omics = torch.randn(1, 48).to(device)
    # test_x = torch.cat([test_img.view(1, -1), test_omics], dim=1)

    # 2. 再取 1 条真实样本做待解释
    # sample = next(iter(train_loader))
    for sample in train_loader:
    # all_samples = list(train_loader)
    # rand_idx = random.randint(0, len(all_samples) - 1)
    # sample = all_samples[rand_idx]
        test_img   = sample['image'][:1].to(device)
        test_omics = sample['omics'][:1].to(device)
        maska= sample['mask_A'][:1].to(device)
        case_id =sample['case_id'][0]
        # print(sample['case_id'][0])
        test_x = torch.cat([test_img.view(1, -1), test_omics], dim=1)
        # ---------------- 建立解释器 -------------------
        explainer = shap.GradientExplainer(wrap_model, bg)
        shap_values = explainer.shap_values(test_x, nsamples=200)

        # 对于二分类，返回 [neg_shap, pos_shap]，我们关心正类
        sv = shap_values[0] # [3*H*W + 48]

        # 拆成图像和组学两部分

        img_total = 3 * sizec * sizec  # 64068
        omics_total = 48           # 48
        total_len = img_total + omics_total

        # 确保背景样本长度正确（若不足补 0，若超则截断）
        bg_np = bg.cpu().numpy()
        bg_np = bg_np[:, :total_len]        # 截断或补零
        test_np = test_x.cpu().numpy()[:, :total_len]
        print(test_np.shape)
        print(sv.shape)

        # ---------------- 拆分 -------------------------
        # 截断到期望长度，再拆分
        sv = sv[:,1]          # 去掉尾部可能的补零
        img_sv = sv[:img_total].reshape(3, sizec, sizec).mean(0)
        maska = maska.detach().cpu().numpy()
        maska = np.broadcast_to(maska, (3, sizec, sizec))
        # print('maska',maska.shape)
        img_sv = img_sv*maska
        omics_sv = sv[img_total:]
        # print('这个是相乘之后',img_sv.shape)


        ##-----------------------画图--------------------------
        plt.figure(figsize=(10,6))

        # 3.1 图像激活图
        gray_map = img_sv.mean(0)  # 在通道维取平均 → (360,360)
        vmin = max(abs(gray_map.min()), abs(gray_map.max()))

        plt.subplot(1, 2, 1)
        im = plt.imshow(gray_map, cmap='seismic', vmin=-vmin, vmax=vmin)
        plt.title("Image SHAP (pixel importance)")
        plt.colorbar(im)
        plt.axis('off')

        # 只读第一行即可
        # feature_names = pd.read_csv(r'F:\RSHS\Omics\Lasso\sum_name.csv', nrows=0).columns.tolist()
        save_path = f'SHAP\Shapley_value_{case_id}.png'
        # 3.2 组学特征条形图
        plt.subplot(1,2,2)
        plt.title("Omics SHAP")
        y = np.arange(len(omics_sv))
        plt.barh(y, omics_sv, color=['red' if v>0 else 'blue' for v in omics_sv])
        plt.gca().set_yticks(y)
        plt.gca().set_yticklabels([f'{i+1}' for i in range(len(omics_sv))],fontsize=8)
        # plt.gca().set_yticklabels(feature_names)  # 用真实名称
        plt.xlabel("Shapley value")
        plt.tight_layout()
        plt.savefig(save_path, dpi=800, bbox_inches='tight')
        # plt.show()
        plt.close()

