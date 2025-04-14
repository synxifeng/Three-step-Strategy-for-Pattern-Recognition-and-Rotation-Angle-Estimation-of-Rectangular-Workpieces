import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import timm
import json
from torch.utils.data import DataLoader

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 40
LR = 1e-4

# Dataset paths
DATA_DIR = r'C:\Users\admin\Desktop\robojigzaw\enhanced_car'
TRAIN_DIR = DATA_DIR + '/train'
VAL_DIR = DATA_DIR + '/val'
TEST_DIR = DATA_DIR + '/test'

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Datasets
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
# Model
model = timm.create_model("swin_base_patch4_window7_224", pretrained=False, num_classes=16)  # set pretrained to False

# Load the weights manually
weights_path = r'C:\Users\admin\Desktop\robojigzaw\pytorch_model.bin'
# Load the weights manually
pretrained_dict = torch.load(weights_path)
model_dict = model.state_dict()

# 1. Filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
# 2. Overwrite entries in the existing state dict
model_dict.update(pretrained_dict)
# 3. Load the new state dict
model.load_state_dict(model_dict)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


# Training loop
for epoch in range(EPOCHS):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 计算测试集的准确率
    test_accuracy = evaluate(model, test_loader)

    # 打印测试集准确率
    print(f"Epoch [{epoch + 1}/{EPOCHS}] Loss: {loss.item()} Accuracy on Test set: {test_accuracy * 100:.2f}%")

# Save model weights
torch.save(model.state_dict(), 'swin_transformer_weights1029.pth')

# Save idx_to_labels mapping
idx_to_labels = train_dataset.class_to_idx
with open('idx_to_labelsswin1029.json', 'w') as f:
    json.dump(idx_to_labels, f)
