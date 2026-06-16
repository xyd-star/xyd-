import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from tqdm import tqdm

# ---------------------- 垃圾分类数据集 ----------------------
class GarbageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.images = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_dir):
                self.images.append((os.path.join(cls_dir, img_name), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# ---------------------- ResNeXt 分类模型 ----------------------
class ResNeXtClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.backbone = models.resnext50_32x4d(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

# ---------------------- 训练主函数 ----------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 训练设备: {device}")

    # 1. 数据增强与预处理
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. 构建数据集（你的真实路径）
    data_root = r"C:\Users\Lenovo\Desktop\test\datasets\Garbage classification"
    full_dataset = GarbageDataset(data_root, transform=train_transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    print(f"📊 数据集大小: 训练集 {train_size}, 验证集 {val_size}")

    # 3. 模型、损失、优化器
    model = ResNeXtClassifier(num_classes=6).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    # 4. TensorBoard 日志
    writer = SummaryWriter(log_dir="runs/garbage_cls")
    best_acc = 0.0
    epochs = 30
    print(f"📦 模型可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 5. 训练循环
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} 训练")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            preds = torch.max(outputs, dim=1)[1]
            train_correct += torch.sum(preds == labels.data)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= len(train_dataset)
        train_acc = train_correct.double() / len(train_dataset)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} 验证")
            for imgs, labels in pbar_val:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = torch.max(outputs, dim=1)[1]
                val_correct += torch.sum(preds == labels.data)

        val_loss /= len(val_dataset)
        val_acc = val_correct.double() / len(val_dataset)

        # 写入日志
        writer.add_scalar("Loss/训练", train_loss, epoch)
        writer.add_scalar("Loss/验证", val_loss, epoch)
        writer.add_scalar("准确率/训练", train_acc, epoch)
        writer.add_scalar("准确率/验证", val_acc, epoch)

        print(f"\n📈 Epoch [{epoch+1:02d}/{epochs}]")
        print(f"训练 Loss: {train_loss:.4f} | 准确率: {train_acc:.4f}")
        print(f"验证 Loss: {val_loss:.4f} | 准确率: {val_acc:.4f}")

        # 保存最优模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_garbage_model.pth")
            print(f"✅ 最优模型已保存，当前最佳验证准确率: {best_acc:.4f}")

        scheduler.step()

    writer.close()
    print(f"\n🏁 训练结束！最高验证准确率: {best_acc:.4f}")

if __name__ == "__main__":
    main()