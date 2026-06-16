import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# 模型定义（必须和训练时一致）
class ResNeXtClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.backbone = models.resnext50_32x4d(pretrained=False)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


# 类别映射
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNeXtClassifier(num_classes=6).to(device)
model.load_state_dict(torch.load("best_garbage_model.pth", map_location=device))
model.eval()

# 预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 预测单张图片
def predict_image(img_path):
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)
        class_name = class_names[pred.item()]
        confidence = torch.softmax(output, dim=1)[0][pred.item()].item()

    return class_name, confidence


# 测试
if __name__ == "__main__":
    img_path = r"C:\Users\Lenovo\Desktop\test\datasets\Garbage classification\cardboard\cardboard1.jpg"
    class_name, confidence = predict_image(img_path)
    print(f"预测类别: {class_name}, 置信度: {confidence:.4f}")