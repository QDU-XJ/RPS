from datetime import datetime
import os
import random
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import torch
from focal_loss import FocalLoss
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model2 import MultiModalClassifier
from dataset4 import MIMDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from itertools import compress
import json
from Modify_data_files.Excel_processing import waiyuan_label
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def concordance_index(y_true, y_pred):
    """计算C-index"""
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()
    concord = 0
    total = 0
    for i in range(len(y_true)):
        for j in range(len(y_true)):
            if i == j:
                continue
            if y_true[i] > y_true[j]:
                total += 1
                if y_pred[i] > y_pred[j]:
                    concord += 1
                elif y_pred[i] == y_pred[j]:
                    concord += 0.5
    return concord / total if total > 0 else 0.5


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, device='cuda',
                model_save_dir='saved_models', seed_num=42):
    """
    训练模型并保存最佳模型

    参数:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        num_epochs: 训练轮数
        device: 训练设备
        model_save_dir: 模型保存目录
        seed_num:当前的随机种子

    返回:
        model: 训练好的模型
        train_loss_history: 训练loss历史
        val_loss_history: 验证loss历史
        metrics_history: 评估指标历史
        best_model_path: 最佳模型保存路径
    """
    # 创建模型保存目录
    os.makedirs(model_save_dir, exist_ok=True)

    model = model.to(device)
    train_loss_history = []
    train_acc_history = []
    val_acc_history = []
    train_auc_history = []
    val_auc_history = []
    val_loss_history = []
    metrics_history = {
        'train_auc': [], 'val_auc': [],
        'train_acc': [], 'val_acc': [],
        'train_cindex': [], 'val_cindex': [],
        'train_loss': [], 'val_loss': [],
        'train_f1': [], 'val_f1': [],
        'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': []
    }

    best_val_acc = 0.3
    best_model_wts = None
    best_epoch = 0
    best_model_path = ''

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # 训练阶段
        model.train()
        running_loss = 0.0
        all_labels = []
        all_probs = []
        all_preds = []

        for batch in train_loader:
            images = batch['image'].to(device).float()
            omics = batch['omics'].to(device).float()
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(images, omics)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().detach())
            all_probs.extend(probs.cpu().detach())
            all_preds.extend(preds.cpu().detach())

        # 计算训练指标
        epoch_loss = running_loss / len(train_loader.dataset)
        train_loss_history.append(epoch_loss)

        train_auc = roc_auc_score(all_labels, all_probs)
        train_acc = accuracy_score(all_labels, all_preds)
        train_cindex = concordance_index(torch.tensor(all_labels), torch.tensor(all_probs))
        train_f1 = f1_score(all_labels, all_preds, average='macro')
        train_precision = precision_score(all_labels, all_preds, average='macro', zero_division=1)
        train_recall = recall_score(all_labels, all_preds, average='macro')
        train_acc_history.append(train_acc)
        train_auc_history.append(train_auc)


        metrics_history['train_auc'].append(train_auc)
        metrics_history['train_acc'].append(train_acc)
        metrics_history['train_cindex'].append(train_cindex)
        metrics_history['train_f1'].append(train_f1)
        metrics_history['train_precision'].append(train_precision)
        metrics_history['train_recall'].append(train_recall)
        metrics_history['train_loss'].append(epoch_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_labels = []
        val_probs = []
        val_preds = []
        wrong_cases = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device).float()
                omics = batch['omics'].to(device).float()
                labels = batch['label'].to(device)
                case_id = batch['case_id']

                outputs = model(images, omics)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds = torch.argmax(outputs, dim=1)

                val_labels.extend(labels.cpu().detach())
                val_probs.extend(probs.cpu().detach())
                val_preds.extend(preds.cpu().detach())

                # 3. 找出预测错误的样本
                mask_wrong = (preds != labels).cpu().numpy()
                wrong_cases.extend(compress(case_id, mask_wrong))  # 对应取出字符串


        # 计算验证指标
        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_loss_history.append(epoch_val_loss)

        val_auc = roc_auc_score(val_labels, val_probs)
        val_acc = accuracy_score(val_labels, val_preds)
        val_cindex = concordance_index(torch.tensor(val_labels), torch.tensor(val_probs))
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        val_precision = precision_score(val_labels, val_preds, average='macro', zero_division=1)
        val_recall = recall_score(val_labels, val_preds, average='macro')

        val_acc_history.append(val_acc)
        val_auc_history.append(val_auc)

        if val_acc > 0.3:
            # 打印/保存所有判别错误的 case_id
            print("判别错误的 case_id（共 {} 个）：".format(len(wrong_cases)))
            lable_list = waiyuan_label(wrong_cases)
            sum1 = sum(lable_list)
            print('其中为1的个数是{} 个'.format(sum1))
            for cid in sorted(wrong_cases):
                print(cid, end=', ')
            print(' ')

        metrics_history['val_auc'].append(val_auc)
        metrics_history['val_acc'].append(val_acc)
        metrics_history['val_cindex'].append(val_cindex)

        # 打印结果
        print(f'Train Loss: {epoch_loss:.4f}            | Val Loss: {epoch_val_loss:.4f}')
        print(f'Train AUC: {train_auc:.4f}              | Val AUC: {val_auc:.4f}')
        print(f'Train Acc: {train_acc:.4f}              | Val Acc: {val_acc:.4f}')
        print(f'Train C-index: {train_cindex:.4f}       | Val C-index: {val_cindex:.4f}')
        print(f'Train F1: {train_f1:.4f}                | Val F1: {val_f1:.4f}')
        print(f'Train Precision: {train_precision:.4f}  | Val Precision: {val_precision:.4f}')
        print(f'Train Recall: {train_recall:.4f}        | Val Recall: {val_recall:.4f}')

        # 保存最佳模型
        if val_acc > best_val_acc and epoch>3:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_model_wts = model.state_dict()

            # 删除旧的最佳模型
            if os.path.exists(best_model_path):
                os.remove(best_model_path)

            # 保存新最佳模型
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            best_model_path = os.path.join(
                model_save_dir,
                f"best_model4_epoch{best_epoch}_acc{val_acc:.4f}_auc{val_auc:.4f}_seed{seed_num}_{timestamp}.pth"
            )
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': best_model_wts,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'val_loss': epoch_val_loss,
            }, best_model_path)
            print(f"New best model saved at: {best_model_path}")
        if epoch<12 and val_acc>0.77:
            break
        if epoch>12 and val_acc<0.22:
            break
        print()

    ## 训练结束后加载最佳模型权重
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
        print(f"Loaded best model from epoch {best_epoch} with val ACC {best_val_acc:.4f}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 绘制loss曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('save_figure/Loss_Train_and_value'+f'seed{seed_num}_ACC {best_val_acc:.4f}_{timestamp}'+'.png')
   ##绘制acc曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_acc_history, label='Train ACC')
    plt.plot(val_acc_history, label='Validation ACC')
    plt.title('Training and Validation acc curve')
    plt.xlabel('Epochs')
    plt.ylabel('ACC')
    plt.legend()
    plt.savefig('save_figure/ACC_train_and_value_' + f'seed{seed_num}_ACC {best_val_acc:.4f}_{timestamp}' + '.png')

    ##绘制AUC曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_auc_history, label='Train AUC')
    plt.plot(val_auc_history, label='Validation AUC')
    plt.title('Training and Validation AUC curve')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend()
    plt.savefig('save_figure/ACC_train_and_value_' + f'seed{seed_num}_AUC {best_val_acc:.4f}_{timestamp}' + '.png')

    # 保存  metrics_history
    json_path = 'saved_models/json'+ f'seed{seed_num}_ACC {best_val_acc:.4f}_{timestamp}.json'
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_history, f, ensure_ascii=False, indent=4)


    return model, train_loss_history, val_loss_history, metrics_history, best_model_path


# 模型加载函数
def load_best_model(model, model_path, optimizer=None, device='cuda'):
    """
    加载保存的最佳模型

    参数:
        model: 模型实例
        model_path: 模型保存路径
        optimizer: 可选，优化器实例
        device: 目标设备

    返回:
        model: 加载权重后的模型
        optimizer: 加载状态后的优化器(如果提供)
        epoch: 保存时的epoch
        val_auc: 保存时的验证AUC
    """
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Loaded model from {model_path}")
    print(f"Previously achieved val AUC: {checkpoint['val_auc']:.4f} at epoch {checkpoint['epoch']}")

    if optimizer is not None:
        return model, optimizer, checkpoint['epoch'], checkpoint['val_auc']
    return model, checkpoint['epoch'], checkpoint['val_auc']

# 使用示例
if __name__ == "__main__":
    # 假设我们已经定义了模型和数据加载器
    # seed_num=[42,2024,564,11026,358] ## 掩膜的多尺度注意力特征提取模块的参数调节，模块A1(model)
    # seed_num = [2025,2024,123, 1990,22,236,636,666,660,21,12,1231,60,862,33334,789,558,202205,202206,202256] ## model3
    # seed_num =[123,558,862,33334,2025,2024,123, 1990,22,236,636,666,660,21,12,1231,60,862,33334,789,558,202205,202206,202256,2003,2004,2020,1998,123,456,789,2024,3369,2027,2030,258,371,1990,2022,506,1991,1968,9393,79]
    seed_num = [666,60,636,3369,2003,2027,96]
    # seed_model = [123]
    # seed_model = [258,371,1990,2022,506,1991,1968,9393,79,24,26,33333] ## 2022的结果是0.6161
    # seed_model =[2003,2004,2020,1998,123,456,789,2024,3369,2027,2030,258,371,1990,2022,506,1991,1968,9393,79]## 其中123--的结果最好为0.6161
    train_transform = A.Compose([
        # A.Rotate(limit=5, border_mode=cv2.BORDER_REFLECT, p=0.2),
        A.ElasticTransform(alpha=2, sigma=50, p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
        A.ColorJitter(brightness=0.1, contrast=0.1, p=0.15),
        # A.HorizontalFlip(p=0.1),
        A.Sharpen(alpha=(0.05, 0.3), lightness=(0.85, 1.15), p=0.25),
        A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.2),
        A.GaussNoise(std_range=(0.1,0.45),mean_range=(-0.15,0.15), p=0.25),
        A.GaussianBlur(blur_limit=[3,5], sigma_limit=(0.5, 2), p=0.15),
        ToTensorV2()
    ])  # mask 与 image 同步几何变换

    # 验证阶段：只做 resize + 归一化
    val_transform = A.Compose([
        # A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.01),
        ToTensorV2()
    ])

    for se in seed_num:
        print('当前种子是：', se)
        set_seed(se)
        model = MultiModalClassifier(se)
        batch_size = 30
        ## 训练数据集
        INPUT_DIR = r"F:\RSHS2\localH_crop"
        exl_path = r'F:\RSHS\Omics\Lasso\n_sum.csv'
        label_excel = r'F:\RSHS2\label.xlsx'
        train_data = MIMDataset(INPUT_DIR, exl_path, label_excel, transform=train_transform)
        ## 测试数据集

        INPUT_DIR = r"F:\RSHS2\OutH_crop"
        # exl_path = r'F:\RSHS\Omics\Lasso\n_sum.csv'
        # label_excel = r'F:\RSHS\label.xlsx'
        # generator = torch.Generator()
        # generator.manual_seed(se)
        test_data = MIMDataset(INPUT_DIR, exl_path, label_excel, transform=val_transform)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
        val_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)

        # 定义损失函数和优化器
        ## 前面的权重是0，后面的是1，也就是0.8 对应的标签0， 1对应的标签是1.8
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.8, 2.6],device='cuda'))
        # criterion = FocalLoss(gamma=2.0, weight=torch.tensor([1.0, 0.6]))
        # optimizer = optim.Adam(model.parameters(), lr=0.0002, weight_decay=1e-3)
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),lr=5e-5, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

        # 训练模型
        trained_model, train_loss, val_loss, metrics, best_path = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=100,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            model_save_dir='model_checkpoints',
            seed_num=se
        )

    # 加载最佳模型
    # loaded_model = load_best_model(model, best_path)