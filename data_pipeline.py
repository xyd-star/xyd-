#  数据工程模块
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# 固定随机种子（保证实验可复现）
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

class GarbageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        self.class_to_idx = {c:i for i,c in enumerate(self.classes)}
        self.images = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_dir):
                self.images.append((os.path.join(cls_dir, img_name), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# 训练集：数据增强 + 归一化
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),   # 随机水平翻转
    transforms.RandomRotation(15),              # 随机旋转±15度
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# 验证集：不做增强，仅标准化
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 数据集路径
data_root = r"C:\Users\Lenovo\Desktop\test\datasets\Garbage classification"
full_dataset = GarbageDataset(data_root, transform=train_transform)

# 划分训练集、验证集
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
val_dataset.dataset.transform = val_transform

# DataLoader完整封装
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

# 打印数据集大小核对
print(f"训练集样本总数：{len(train_dataset)}")
print(f"验证集样本总数：{len(val_dataset)}")


# ---------------------- 1.2 类别分布可视化 ----------------------
def plot_class_distribution(dataset):
    # 统计每个类别的样本数量
    class_counts = {c: 0 for c in dataset.classes}
    for _, label in dataset.images:
        class_counts[dataset.classes[label]] += 1

    # 绘制柱状图
    plt.figure(figsize=(8, 5))
    bars = plt.bar(class_counts.keys(), class_counts.values(), color='#1f77b4')

    # 在柱子上标注具体数量
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}',
                 ha='center', va='bottom')

    plt.title("Garbage Classification Dataset Class Distribution", fontsize=12)
    plt.xlabel("Garbage Class", fontsize=10)
    plt.ylabel("Number of Images", fontsize=10)
    plt.xticks(rotation=30)  # 旋转x轴标签避免重叠
    plt.tight_layout()  # 自动调整布局

    # 保存图片（报告里直接用这张图）
    plt.savefig("garbage_class_distribution.png", dpi=300)
    plt.show()


# 调用可视化函数（传入完整数据集）
plot_class_distribution(full_dataset)