import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
from model2 import MultiModalClassifier  # 你的网络
from dataset4 import MIMDataset  # 你的数据集类

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. 数据集 / DataLoader
INPUT_DIR = r"F:\RSHS2\localH_crop"
exl_path = r'F:\RSHS\Omics\Lasso\n_sum.csv'
label_excel = r'F:\RSHS2\label.xlsx'

val_set = MIMDataset(INPUT_DIR, exl_path, label_excel)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)

# 2. 模型
model = MultiModalClassifier().to(device)
ckpt = torch.load('model_checkpoints/best_model.pth', map_location=device)
state = ckpt.get('model_state_dict', ckpt)
model.load_state_dict(state)
model.eval()

# 存储所有预测结果
all_case_ids = []
all_probs = []
all_labels = []

# 3. 收集预测结果
with torch.no_grad():
    for batch in val_loader:
        img = batch['image'].to(device).float()
        omics = batch['omics'].to(device).float()
        cid = batch['case_id'][0]  # 字符串
        label = batch['label'].item()  # 真实标签

        logits = model(img, omics)
        prob = torch.softmax(logits, dim=1)[0, 1].item()  # 正类概率

        all_case_ids.append(cid)
        all_probs.append(prob)
        all_labels.append(label)


# 4. 计算AUC和绘制ROC曲线
def calculate_auc_and_plot_roc(labels, probs, save_path='roc_curve.png'):
    """
    计算AUC并绘制ROC曲线

    Args:
        labels: 真实标签列表
        probs: 预测概率列表
        save_path: ROC曲线保存路径
    """
    # 转换为numpy数组
    labels = np.array(labels)
    probs = np.array(probs)

    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    # 绘制ROC曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)

    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    return roc_auc, fpr, tpr, thresholds


# 5. 计算AUC并绘制ROC曲线
auc_value, fpr, tpr, thresholds = calculate_auc_and_plot_roc(all_labels, all_probs)

print(f"AUC: {auc_value:.4f}")

# 6. 输出预测概率到CSV文件（包含真实标签和预测结果）
with open('train_probs_with_labels.csv', 'w') as f:
    f.write('case_id,prob,true_label\n')
    for case_id, prob, label in zip(all_case_ids, all_probs, all_labels):
        f.write(f"{case_id},{prob:.4f},{label}\n")


# 7. 可选：输出详细的性能指标
def print_detailed_metrics(labels, probs, threshold=0.5):
    """
    打印详细的分类性能指标
    """
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

    predictions = (np.array(probs) >= threshold).astype(int)
    labels = np.array(labels)

    print("\n" + "=" * 50)
    print("详细性能指标")
    print("=" * 50)
    print(f"阈值: {threshold}")
    print(f"准确率: {accuracy_score(labels, predictions):.4f}")
    print("\n混淆矩阵:")
    cm = confusion_matrix(labels, predictions)
    print(cm)
    print("\n分类报告:")
    print(classification_report(labels, predictions, target_names=['Class 0', 'Class 1']))

    # 计算敏感性和特异性
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"敏感性 (召回率): {sensitivity:.4f}")
    print(f"特异性: {specificity:.4f}")


# 打印详细指标
print_detailed_metrics(all_labels, all_probs)


# 8. 可选：找到最佳阈值
def find_optimal_threshold(fpr, tpr, thresholds):
    """
    根据Youden指数找到最佳阈值
    """
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold


optimal_threshold = find_optimal_threshold(fpr, tpr, thresholds)
print(f"\n最佳阈值 (Youden指数): {optimal_threshold:.4f}")

# 使用最佳阈值重新计算指标
print("\n使用最佳阈值的性能指标:")
print_detailed_metrics(all_labels, all_probs, optimal_threshold)