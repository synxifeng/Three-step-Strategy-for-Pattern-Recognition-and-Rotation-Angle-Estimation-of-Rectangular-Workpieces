# import os
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# from PIL import Image
#
# DATA_DIR = r'C:\Users\xifeng\Desktop\robojigzaw\car'
# TRAIN_DIR = os.path.join(DATA_DIR, 'train')
# VAL_DIR = os.path.join(DATA_DIR, 'val')
#
# # 定义数据增强转换
# transform = transforms.Compose([
#     transforms.RandomRotation(30),  # 随机旋转图像，角度在[-30, 30]之间
#     transforms.RandomResizedCrop(224, scale=(0.8, 1.2)),  # 随机放大缩小图像并裁剪到224x224
#     transforms.RandomHorizontalFlip(),  # 随机水平翻转图像
# ])
#
# # 使用定义的转换来加载数据集
# train_dataset = datasets.ImageFolder(TRAIN_DIR)
# val_dataset = datasets.ImageFolder(VAL_DIR)
#
# # 保存增强后的图像的目录
# SAVE_DIR = r'C:\Users\xifeng\Desktop\robojigzaw\enhanced_data'
# TRAIN_SAVE_DIR = os.path.join(SAVE_DIR, 'train')
# VAL_SAVE_DIR = os.path.join(SAVE_DIR, 'val')
#
# NUM_AUGMENTATIONS = 4
#
#
# def save_augmented_images(dataset, save_dir):
#     for (image, label) in dataset:
#         class_dir = os.path.join(save_dir, dataset.classes[label])
#         if not os.path.exists(class_dir):
#             os.makedirs(class_dir)
#
#         base_name = os.path.splitext(os.path.basename(dataset.samples[label][0]))[0]
#         for i in range(NUM_AUGMENTATIONS):
#             augmented_image = transform(image)
#             augmented_image_path = os.path.join(class_dir, f"{base_name}_aug_{i}.jpg")
#             augmented_image.save(augmented_image_path)
#
#
# save_augmented_images(train_dataset, TRAIN_SAVE_DIR)
# save_augmented_images(val_dataset, VAL_SAVE_DIR)


import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image

DATA_DIR = r'C:\Users\admin\Desktop\car'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# 定义数据增强转换
transform = transforms.Compose([
    transforms.RandomRotation(30),  # 随机旋转图像，角度在[-30, 30]之间
    transforms.RandomResizedCrop(224, scale=(0.8, 1.2)),  # 随机放大缩小图像并裁剪到224x224
    transforms.RandomHorizontalFlip(),  # 随机水平翻转图像
])

# 使用定义的转换来加载数据集
train_dataset = datasets.ImageFolder(TRAIN_DIR)
val_dataset = datasets.ImageFolder(VAL_DIR)
test_dataset = datasets.ImageFolder(TEST_DIR)
# 保存增强后的图像的目录
SAVE_DIR = r'C:\Users\admin\Desktop\robojigzaw\enhanced_car'
TRAIN_SAVE_DIR = os.path.join(SAVE_DIR, 'train')
VAL_SAVE_DIR = os.path.join(SAVE_DIR, 'val')
TEST_SAVE_DIR = os.path.join(SAVE_DIR, 'test')
NUM_AUGMENTATIONS = 10


def save_augmented_images(dataset, save_dir):
    for idx, (image, label) in enumerate(dataset):
        class_dir = os.path.join(save_dir, dataset.classes[label])
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        base_name = os.path.splitext(os.path.basename(dataset.samples[idx][0]))[0]
        for i in range(NUM_AUGMENTATIONS):
            augmented_image = transform(image)
            augmented_image_path = os.path.join(class_dir, f"{base_name}_aug_{i}.jpg")
            augmented_image.save(augmented_image_path)


save_augmented_images(train_dataset, TRAIN_SAVE_DIR)
save_augmented_images(val_dataset, VAL_SAVE_DIR)
save_augmented_images(test_dataset, TEST_SAVE_DIR)