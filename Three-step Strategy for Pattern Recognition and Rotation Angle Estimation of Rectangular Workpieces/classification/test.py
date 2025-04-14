# import torch
# import timm
# from torchvision import transforms
# from PIL import Image
# import json
#
#
# # Load the pre-trained model architecture
# model = timm.create_model("swin_base_patch4_window7_224", pretrained=False, num_classes=16)
#
# # Load the trained weights
# weights_path = 'swin_transformer_weights1029.pth'  # Update with the path to your saved model weights
# model.load_state_dict(torch.load(weights_path))
# model.eval()
#
#
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
#
# # Load and preprocess your input image
# image_path = r'C:\Users\admin\Desktop\baizhengcar\16_reference.jpg'  # Update with the path to your input image
# input_image = Image.open(image_path)
# input_tensor = transform(input_image).unsqueeze(0)  # Add an extra dimension for batch size
#
# # Ensure your model is in evaluation mode
# model.eval()
#
# # Forward pass
# with torch.no_grad():
#     output = model(input_tensor)
#
# # Get the predicted class
# _, predicted_class = output.max(1)
#
# # Load idx_to_labels mapping
# with open('idx_to_labelsswin1029.json', 'r') as f:
#     idx_to_labels = json.load(f)
#
# # Get the predicted label
# predicted_label = idx_to_labels[str(predicted_class.item())]
#
# print(f"Predicted class: {predicted_label}")


import torch
import timm
from torchvision import transforms
from PIL import Image
import json
import time

# Load the pre-trained model architecture
model = timm.create_model("swin_base_patch4_window7_224", pretrained=False, num_classes=16)

# Load the trained weights
weights_path = 'swin_transformer_weights1029.pth'  # Update with the path to your saved model weights
model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and preprocess your input image
image_path = r'C:\Users\admin\Desktop\baizhengcar\16_reference.jpg'  # Update with the path to your input image
input_image = Image.open(image_path)
input_tensor = transform(input_image).unsqueeze(0)  # Add an extra dimension for batch size

# Ensure the model is in evaluation mode
model.eval()

# Measure time taken for the forward pass
start_time = time.time()
with torch.no_grad():
    output = model(input_tensor)
end_time = time.time()

# Calculate and print the execution time
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.4f} seconds")

# Get the predicted class
_, predicted_class = output.max(1)

# Load the idx_to_labels mapping
with open('idx_to_labelsswin1029.json', 'r') as f:
    idx_to_labels = json.load(f)

# Get the predicted label
predicted_label = idx_to_labels[str(predicted_class.item())]

print(f"Predicted class: {predicted_label}")
