import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
import time
from data_pipeline import train_loader, val_loader, train_dataset, val_dataset


# 模型定义（和训练时一致）
class ResNeXtClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.backbone = models.resnext50_32x4d(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


# ---------------------- 三组超参数配置 ----------------------
exp_configs = [
    {
        "exp_name": "exp1_lr1e-4_Adam_b16",
        "lr": 1e-4,
        "optimizer": "Adam",
        "batch_size": 16,
        "epochs": 30
    },
    {
        "exp_name": "exp2_lr1e-5_Adam_b16",
        "lr": 1e-5,
        "optimizer": "Adam",
        "batch_size": 16,
        "epochs": 30
    },
    {
        "exp_name": "exp3_lr1e-4_SGD_b32",
        "lr": 1e-4,
        "optimizer": "SGD",
        "batch_size": 32,
        "epochs": 30
    }
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results = []

# ---------------------- 循环训练三组实验 ----------------------
for cfg in exp_configs:
    print(f"\n{'=' * 50}")
    print(f"开始训练：{cfg['exp_name']}")
    print(f"{'=' * 50}")

    # 初始化模型和优化器
    model = ResNeXtClassifier(num_classes=6).to(device)
    criterion = nn.CrossEntropyLoss()

    if cfg["optimizer"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    else:
        optimizer = optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    writer = SummaryWriter(log_dir=f"runs/{cfg['exp_name']}")
    best_val_acc = 0.0
    epoch_times = []

    for epoch in range(cfg["epochs"]):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        t_start = time.time()

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()

        t_cost = time.time() - t_start
        epoch_times.append(t_cost)
        train_loss /= len(train_dataset)
        train_acc = train_correct / len(train_dataset)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
        val_loss /= len(val_dataset)
        val_acc = val_correct / len(val_dataset)

        # 写入TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Acc/val", val_acc, epoch)
        writer.flush()

        # 保存最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{cfg['exp_name']}_best.pth")

        print(
            f"Epoch {epoch + 1:2d} | TrainLoss:{train_loss:.4f} TrainAcc:{train_acc:.4f} | ValLoss:{val_loss:.4f} ValAcc:{val_acc:.4f} | Time:{t_cost:.1f}s")
        scheduler.step()

    writer.close()
    avg_time = sum(epoch_times) / len(epoch_times)
    results.append({
        "exp_name": cfg["exp_name"],
        "lr": cfg["lr"],
        "optimizer": cfg["optimizer"],
        "batch_size": cfg["batch_size"],
        "best_val_acc": best_val_acc,
        "avg_epoch_time": avg_time
    })
    print(f"\n【{cfg['exp_name']}】训练完成！最佳验证准确率：{best_val_acc:.4f}，平均每轮耗时：{avg_time:.1f}s")

# ---------------------- 输出对比表格 ----------------------
print(f"\n{'=' * 70}")
print("三组超参数对比实验结果汇总")
print(f"{'=' * 70}")
print(
    f"{'实验名称':<25} | {'学习率':<8} | {'优化器':<5} | {'BatchSize':<10} | {'最佳验证准确率':<15} | {'平均每轮耗时(s)':<15}")
print("-" * 70)
for res in results:
    print(
        f"{res['exp_name']:<25} | {res['lr']:.1e}    | {res['optimizer']:<5} | {res['batch_size']:<10} | {res['best_val_acc']:.4f}         | {res['avg_epoch_time']:.1f}")