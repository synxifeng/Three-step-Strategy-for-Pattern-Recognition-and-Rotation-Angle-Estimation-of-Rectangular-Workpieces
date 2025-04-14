# import cv2
# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# import numpy as np
# # 加载mobilenet_v2
# from torchvision.models.mobilenet import mobilenet_v2
#
# # 加载MobileNetV2
# model_path = 'mobilenet0617_car.pth'
# model = mobilenet_v2(pretrained=False)
#
# # 替换全连接层以匹配检查点中的权重尺寸
# num_classes = 16  # 将这个值设置为检查点中全连接层的输出尺寸
# num_ftrs = model.classifier[1].in_features
# model.classifier[1] = torch.nn.Linear(num_ftrs, num_classes)
#
# # 加载模型权重
# model.load_state_dict(torch.load(model_path))
# model.eval()
#
# # 加载类别索引号对应列表
# idx_to_labels = np.load('idx_class_car0617.npy', allow_pickle=True).item()
#
# # 图像预处理
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
#
# # 定义函数，用于分类物体
# def classify_object(image_path, model):
#     image = cv2.imread(image_path)
#     img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     img_tensor = transform(img_pil).unsqueeze(0)
#     with torch.no_grad():
#         output = model(img_tensor)
#         _, preds = torch.max(output, 1)
#         predicted_class = idx_to_labels[preds.item()]
#         return predicted_class
#
# # 示例用法
# image_path =  r'C:\Users\admin\Desktop\robojigzaw\1021shuju\2_34j.jpg'
# predicted_label = classify_object(image_path, model)
# print(f"Predicted Label: {predicted_label}")

import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import time

# 加载mobilenet_v2
from torchvision.models.mobilenet import mobilenet_v2

# 加载MobileNetV2
model_path = 'mobilenet0617_car.pth'
model = mobilenet_v2(pretrained=False)

# 替换全连接层以匹配检查点中的权重尺寸
num_classes = 16  # 将这个值设置为检查点中全连接层的输出尺寸
num_ftrs = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_ftrs, num_classes)

# 加载模型权重
model.load_state_dict(torch.load(model_path))
model.eval()

# 加载类别索引号对应列表
idx_to_labels = np.load('idx_class_car0617.npy', allow_pickle=True).item()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 定义函数，用于分类物体
def classify_object(image_path, model):
    image = cv2.imread(image_path)
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img_tensor = transform(img_pil).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        _, preds = torch.max(output, 1)
        predicted_class = idx_to_labels[preds.item()]
        return predicted_class

# 测试用法
image_path =  r'C:\Users\admin\Desktop\robojigzaw\1021shuju\2_34j.jpg'

start_time = time.time()
predicted_label = classify_object(image_path, model)
end_time = time.time()
execution_time = end_time - start_time

print(f"Predicted Label: {predicted_label}")
print(f"Execution Time: {execution_time} seconds")

