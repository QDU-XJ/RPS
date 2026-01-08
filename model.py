import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights

class ResidualAttention(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(channel, channel//4, 3, padding=1),
            nn.BatchNorm2d(channel//4),
            nn.ReLU(),
            nn.Conv2d(channel//4, 1, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x*0.5 + x * self.attn(x)  # 残差连接

class ImageAttentionModule(nn.Module):
    def __init__(self, in_channels=3, out_channels=16):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Mask特征提取分支（保持输入尺寸）模块A1参数
        self.mask_feature_extract = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2)),
            # nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, out_channels, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 多尺度空间注意力机制（统一输出尺寸）
        self.ms_attention = nn.ModuleList([
            # 分支1: 全局注意力（通过自适应池化统一尺寸）
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels, out_channels, kernel_size=1),
                nn.Sigmoid()
            ),
            # 分支2: 3x3卷积（保持尺寸）
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.Sigmoid()
            ),
            # 分支3:7x7卷积（保持尺寸）
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2),
                nn.Sigmoid()
            )
        ])

        # 自适应池化确保最终输出尺寸
        self.adaptive_pool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, image, mask):
        # 确保mask扩展为3通道
        # mask = mask +0.05
        new_mask = mask.expand(-1, 3, -1, -1)

        # 提取mask区域特征
        masked_features = image * new_mask
        y = self.mask_feature_extract(masked_features)  # [B, C, H, W]

        # 获取当前特征图尺寸
        b, c, h, w = y.size()

        # 多尺度注意力计算（确保尺寸匹配）
        attn1 = self.ms_attention[0](y)  # [B, C, 1, 1]
        attn1 = F.interpolate(attn1, size=(h, w), mode='bilinear', align_corners=False)

        attn2 = self.ms_attention[1](y)  # [B, C, H, W]
        attn3 = self.ms_attention[2](y)  # [B, C, H, W]

        # 注意力融合（尺寸已统一）
        combined_attention = (attn1 + attn2 + attn3) / 3.0

        # 应用注意力
        y = y * combined_attention

        # 输出尺寸标准化
        y = self.adaptive_pool(y)
        return y

class MultiModalClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # 图像处理分支 (使用预训练ResNet34)
        weights = ResNet34_Weights.DEFAULT
        self.resnet = resnet34(weights=weights)
        # 冻结 ResNet34 所有参数
        for param in self.resnet.parameters():
            param.requires_grad = False
        # 移除最后的全连接层
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])

        self.resnet_adjust = nn.Sequential(
            nn.Conv2d(512, 12, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, 1))
        )

        # 注意力模块
        self.image_attention = ImageAttentionModule(in_channels=3, out_channels=32)

        # ## 残差模块
        self.resattention = ResidualAttention(channel=12+32)

        # 组学特征处理分支
        self.omics_fc = nn.Sequential(
            nn.Linear(48, 8),
            nn.LayerNorm(8),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(32+12+8, 12),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # 分类器
        self.classifier = nn.Linear(12, 2)

    def forward(self, image, omics, mask):
        # 确保输入数据类型为float32
        image = image.float()
        omics = omics.float()
        mask = mask.float()

        # 确保图像维度正确 [B, C, H, W]
        if image.dim() == 4 and image.size(1) != 3:
            image = image.permute(0, 3, 1, 2)

        # ResNet分支处理
        x_img = self.resnet(image)  # [B, 512, H, W]
        x_img = self.resnet_adjust(x_img)  # [B, 64, 4, 4]

        # 注意力分支处理
        y_img = self.image_attention(image, mask)  # [B, 64, 4, 4]

        # 合并特征并降维
        features = torch.cat([x_img, y_img], dim=1)
        features = self.resattention(features)

        features = features.reshape(features.size(0), -1)
        # 组学特征处理
        x_omics = self.omics_fc(omics)

        # 特征融合
        x = torch.cat([features, x_omics], dim=1)
        x = self.fusion(x)

        # 分类
        out = self.classifier(x)

        return out


# 测试模型
if __name__ == "__main__":
    model = MultiModalClassifier()

    # 创建测试输入
    image = torch.randn(2, 3, 146, 146)
    omics = torch.randn(2, 48)
    mask = torch.randn(2, 1, 146, 146)

    # 前向传播
    output = model(image, omics, mask)
    print(f"Output shape: {output.shape}")  # 应该输出 [2, 2]