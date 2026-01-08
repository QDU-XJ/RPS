import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# set_seed(42)

# 训练阶段变换：保留原有配置，但后续手动处理掩码维度
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, border_mode=cv2.BORDER_REFLECT, p=0.7),
    A.ElasticTransform(alpha=1, sigma=50, p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
    # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()  # 仅能保证image转置为(3,H,W)，mask可能仍为(H,W,3)
])


val_transform = A.Compose([
    # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


class MIMDataset(Dataset):
    def __init__(self, image_dir, omics_csv, label_csv_path, transform=None):
        self.image_dir = image_dir
        self.label_pd = pd.read_excel(label_csv_path, dtype={'ID': str}).set_index('ID')['FNCLCC']
        self.omics_pd = pd.read_csv(omics_csv, dtype={'case_id': str}).set_index('case_id')
        self.data_list = [x for x in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, x))]
        self.transform = transform
        # set_seed(42)

    def __len__(self):
        return len(self.data_list)

    def _load_gray(self, path):
        """加载灰度图为 (H, W) 的 numpy 数组，范围 [0,1]"""
        img = Image.open(path).convert('L')
        if img is None:
            raise FileNotFoundError(f"未找到图像文件: {path}")
        img = img.resize((256, 256))
        return np.array(img).astype(np.float32) / 255.0

    def __getitem__(self, idx):
        case_id = self.data_list[idx]
        root = os.path.join(self.image_dir, case_id)

        # 1. 加载三通道图像 (H, W, 3)
        try:
            a = self._load_gray(os.path.join(root, 'A_ORI.png'))
            d = self._load_gray(os.path.join(root, 'D_ORI.png'))
            v = self._load_gray(os.path.join(root, 'V_ORI.png'))
        except FileNotFoundError as e:
            raise FileNotFoundError(f"样本 {case_id} 缺失图像文件: {e}")
        image = np.stack([a, d, v], axis=-1)  # (H, W, 3)

        # 2. 加载三通道掩码 (H, W, 3)
        try:
            mask_A = self._load_gray(os.path.join(root, 'A_MASK.png'))
        #     mask_D = self._load_gray(os.path.join(root, 'D_MASK.png'))
        #     mask_V = self._load_gray(os.path.join(root, 'V_MASK.png'))
        except FileNotFoundError as e:
            raise FileNotFoundError(f"样本 {case_id} 缺失掩码文件: {e}")
        # mask = np.stack([mask_A, mask_D, mask_V], axis=-1)  # (H, W, 3)

        # 3. 验证原始尺寸一致
        # if image.shape != mask.shape:
        #     raise ValueError(
        #         f"样本 {case_id} 原始尺寸不匹配: "
        #         f"图像 {image.shape}，掩码 {mask.shape}"
        #     )

        # 4. 加载标签和omics
        try:
            label = np.array(self.label_pd.loc[case_id], dtype=np.int64)
            omics = np.array(self.omics_pd.loc[case_id].values, dtype=np.float32)
        except KeyError as e:
            raise KeyError(f"样本 {case_id} 未找到标签/omics: {e}")

        # 5. 应用增强 + 核心修复：手动转置掩码维度
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']  # 已转置为 (3, H, W)
            # mask = transformed['mask']  # 此时可能是 (H, W, 3)
            # if mask.ndim == 3 and mask.shape[2] == 3:  # 确认是通道在后的格式
            #     mask = mask.permute(2, 0, 1)  # 转置维度：(H,W,3) → (3,H,W)



        # 7. 应用掩码（类型匹配 + 逐元素相乘）
        # mask = mask.to(dtype=image.dtype)
        # mask = (mask > 0.5).float()  # 转为0或1的二值张量
        # masked_image = image * mask

        return {
            'image': image,
            'omics': torch.from_numpy(omics),
            'label': torch.tensor(label),
            'mask_A': mask_A,
            'case_id': case_id
        }


if __name__ == '__main__':
    INPUT_DIR = r"F:\RSHS2\localH_crop"
    OMICS_CSV = r"F:\RSHS\Omics\Lasso\n_sum.csv"
    LABEL_EXCEL = r"F:\RSHS2\label.xlsx"

    # 数据集和加载器（调试模式）
    train_ds = MIMDataset(INPUT_DIR, OMICS_CSV, LABEL_EXCEL, transform=train_transform)
    train_loader = DataLoader(
        train_ds,
        batch_size=2,  # 最小batch_size，降低调试难度
        shuffle=True,
        num_workers=0,  # 单进程，报错定位更准
        pin_memory=True
    )

    os.makedirs('SHAP', exist_ok=True)

    # 测试1个batch
    for epoch in range(1):
        print(f"Epoch {epoch + 1}")
        for batch_idx, data in enumerate(train_loader):
            print(f"Batch {batch_idx} 图像形状: {data['image'].shape}")

            # 反归一化显示
            # mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            # std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

            # 取第1个样本、第1个通道
            sample_img = data['image'][0, 0, :, :]
            # sample_img_denorm = sample_img * std[0, 0, 0] + mean[0, 0, 0]
            sample_img_np = (sample_img.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

            #mask image
            # sample_mask = data['mask'][0, 0, :, :]
            # sample_mask_np = (sample_mask.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

            # 保存
            save_path = os.path.join('SHAP', f"{data['case_id'][0]}_CROP_image.png")
            Image.fromarray(sample_img_np).save(save_path)
            print(f"已保存: {save_path}")

            # save_path2 = os.path.join('SHAP', f"{data['case_id'][0]}_mask.png")
            # Image.fromarray(sample_mask_np).save(save_path2)
            # print(f"已保存: {save_path}")


            break  # 只跑1个batch
        break