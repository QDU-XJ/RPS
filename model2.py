import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights,densenet121, DenseNet121_Weights
import os

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# set_seed(2024)

class MultiHeadFusion(nn.Module):
    def __init__(self, d_model=24, nhead=4, dropout=0.3):
        super().__init__()
        self.d_model = d_model
        # 不再要 proj_* 层
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed1 = nn.Parameter(torch.randn(1, 3, d_model))
        self.pos_embed2 = nn.Parameter(torch.randn(1, 3, d_model))
        self.pos_embed3 = nn.Parameter(torch.randn(1, 3, d_model))

        self.mha = nn.MultiheadAttention(embed_dim=d_model,
                                         num_heads=nhead,
                                         batch_first=True,
                                         dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,  att_feat, omics):
        B = att_feat.size(0)

        # 直接 unsqueeze
        # res = res_feat.unsqueeze(1)      # [B, 1, 32]
        att = att_feat.unsqueeze(1)
        omi = omics.unsqueeze(1)

        tokens = torch.cat([att, omi], dim=1)      # [B, 3, 32]
        cls_tokens = self.cls_token.expand(B, -1, -1)   # [B, 1, 32]
        seq = torch.cat([cls_tokens, tokens], dim=1)    # [B, 4, 32]

        # seq = seq + self.pos_embed                       # 位置编码

        # Self-Attention
        out, _ = self.mha(seq+self.pos_embed1, seq+self.pos_embed2,seq+self.pos_embed3)
        out = self.norm(out+0.7*seq)                       # 残差 + LN

        # 取 CLS token
        fused = out[:, 0, :]                             # [B, d_model]
        return fused



class MultiModalClassifier(nn.Module):
    def __init__(self,seed=42):
        super().__init__()
        # 图像处理分支 (使用预训练ResNet34)
        set_seed(seed)
        weights = ResNet34_Weights.DEFAULT
        self.resnet = resnet34(weights=weights)
        # weights = DenseNet121_Weights.DEFAULT
        # self.resnet = densenet121(weights=weights)
        # 冻结 ResNet34 所有参数
        for param in self.resnet.parameters():
            param.requires_grad = False
        # 冻结部分底层参数（根据数据集大小调整）
        # for param in list(self.resnet.parameters())[:-3]:  # 只微调最后3层
        #     param.requires_grad = False

        # 移除最后的全连接层
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        fusion_size = 12
        self.resnet_adjust = nn.Sequential(
            nn.Conv2d(512, fusion_size, kernel_size=3,stride=1),
            nn.BatchNorm2d(fusion_size),
            # nn.Dropout(0.1),
            nn.ReLU(),
            # nn.Conv2d(64, fusion_size, kernel_size=1, stride=1),
            # nn.BatchNorm2d(fusion_size),
            # nn.Dropout(0.1),
            # nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, 1))
        )


        # 组学特征处理分支
        self.omics_fc = nn.Sequential(
            nn.Linear(48, fusion_size),
            nn.LayerNorm(fusion_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # ## 多头注意力机制融合
        self.fusion_attn = MultiHeadFusion(d_model=fusion_size, nhead=6, dropout=0.2)
        # 融合层
        # self.fusion = nn.Sequential(
        #     nn.Linear(16, 12),
        #     nn.ReLU(),
        #     nn.Dropout(0.3)
        # )

        # 分类器
        self.classifier = nn.Linear(fusion_size, 2)

    def forward(self, image, omics):
        # 确保输入数据类型为float32
        image = image.float()
        omics = omics.float()

        # 确保图像维度正确 [B, C, H, W]
        if image.dim() == 4 and image.size(1) != 3:
            image = image.permute(0, 3, 1, 2)

        # ResNet分支处理
        x_img = self.resnet(image)  # [B, 512, H, W]
        # print(x_img.size())
        x_img = self.resnet_adjust(x_img)  # [B, 64, 4, 4]


        # 组学特征处理
        x_omics = self.omics_fc(omics)
        # ---- 多头注意力融合 ----
        # 统一压平到 [B, C]
        x_img = x_img.squeeze(-1).squeeze(-1)  # [B, 24]
        # ---- 多头注意力融合 ----
        fused = self.fusion_attn(x_img, x_omics)


        # 分类
        out = self.classifier(fused)

        return out

class WrapperModel(nn.Module):
    """
    把 (image, omics) 拼成一个大张量，方便 SHAP 处理。
    假设 image: [B,3,H,W], omics: [B,48]
    拼接后:   [B, 3*H*W + 48]
    """
    def __init__(self, base: MultiModalClassifier, img_size=(146, 146)):
        super().__init__()
        self.base = base
        self.img_size = img_size
        self.C, self.H, self.W = 3, *img_size

    def forward(self, x):
        # x: [B, 3*H*W + 48]
        B = x.size(0)
        img_flat, omics = x.split([self.C * self.H * self.W, 48], dim=1)
        img = img_flat.view(B, self.C, self.H, self.W)
        return self.base(img, omics)

# 测试模型
if __name__ == "__main__":
    model = MultiModalClassifier()

    # 创建测试输入
    image = torch.randn(2, 3, 256, 256)
    omics = torch.randn(2, 48)
    mask = torch.randn(2, 1, 360, 360)

    # 前向传播
    output = model(image, omics)
    print(f"Output shape: {output.shape}")  # 应该输出 [2, 2]