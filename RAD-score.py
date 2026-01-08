import torch
from torch.utils.data import DataLoader
from itertools import compress
from model2 import MultiModalClassifier     # 你的网络
from dataset4 import MIMDataset                  # 你的数据集类

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. 数据集 / DataLoader
# val_path   = r'F:\李天会\Her2new\Omics60\test.csv'   # 验证集 csv
# img_root   = r'F:\李天会\Her2new\外院2'
# label_excel= r'F:\李天会\Her2new\省立.xlsx'


# INPUT_DIR = r"F:\RSHS2\OutH_crop"
INPUT_DIR = r"F:\RSHS2\localH_crop"
exl_path = r'F:\RSHS\Omics\Lasso\n_sum.csv'
label_excel = r'F:\RSHS2\label.xlsx'

val_set = MIMDataset(INPUT_DIR, exl_path, label_excel)
# val_set   = MIMDataset(img_root, val_path, label_excel)
val_loader= DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)

# 2. 模型
model = MultiModalClassifier().to(device)
ckpt  = torch.load('model_checkpoints/best_model.pth', map_location=device)
state = ckpt.get('model_state_dict', ckpt)
model.load_state_dict(state)
model.eval()

# 3. 输出 (case_id, prob)
with open('train_probs.csv', 'w') as f:
# with open('val_probs.csv', 'w') as f:
    f.write('case_id,prob\n')
    with torch.no_grad():
        for batch in val_loader:
            img   = batch['image'].to(device).float()
            omics = batch['omics'].to(device).float()
            cid   = batch['case_id'][0]          # 字符串

            logits = model(img, omics)
            prob   = torch.softmax(logits, dim=1)[0, 1].item()  # 正类概率
            f.write(f"{cid},{prob:.4f}\n")
