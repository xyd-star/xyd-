import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 模型定义（和训练时完全一致）
class ResNeXtClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.backbone = models.resnext50_32x4d(weights=None)  # 新版写法消除警告
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

# 类别名称映射
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNeXtClassifier(num_classes=6).to(device)
model.load_state_dict(torch.load("best_garbage_model.pth", map_location=device))
model.eval()

# 图片预处理（和训练时保持一致）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 预测函数
def predict_image(img):
    image = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)
        class_name = class_names[pred.item()]
        confidence = torch.softmax(output, dim=1)[0][pred.item()].item()
    return {class_name: float(confidence)}

# 启动 Gradio 界面（端口改为 7861，避免占用）
demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="上传垃圾图片"),
    outputs=gr.Label(label="预测结果"),
    title="垃圾分类识别系统",
    description="上传一张垃圾图片，模型将识别其类别（cardboard/glass/metal/paper/plastic/trash）"
)
demo.launch(server_name="127.0.0.1", server_port=7861)