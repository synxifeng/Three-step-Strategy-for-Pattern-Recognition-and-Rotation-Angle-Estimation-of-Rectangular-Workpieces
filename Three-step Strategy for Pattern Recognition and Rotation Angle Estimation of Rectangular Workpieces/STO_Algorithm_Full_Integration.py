import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import serial
import time
import random
# 加载mobilenet_v2
from torchvision.models.mobilenet import mobilenet_v2
# 加载MobileNetV2
model_path = r'C:\Users\Robot\Desktop\python\mobilenet0617_car.pth'
model = mobilenet_v2(pretrained=False)
# 替换全连接层以匹配检查点中的权重尺寸
num_classes = 16  # 将这个值设置为检查点中全连接层的输出尺寸num_ftrs = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_ftrs, num_classes)
# 加载模型权重
model.load_state_dict(torch.load(model_path))
model.eval()

# # 加载类别索引号对应列表
idx_to_labels = np.load('C:\\Users\Robot\Desktop\python\\idx_class_car0617.npy', allow_pickle=True).item()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def classify_object(image, model):
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img_tensor = transform(img_pil).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        _, preds = torch.max(output, 1)

        return idx_to_labels[preds.item()]

def calculate_rotation_angle(template, actual_image):
    # 创建ORB对象
    orb = cv2.ORB_create()

    # 对两个图像进行关键点检测和描述符计算
    kp1, des1 = orb.detectAndCompute(template, None)
    kp2, des2 = orb.detectAndCompute(actual_image, None)

    # 创建Brute Force Matcher对象
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # 对描述符进行匹配
    matches = bf.match(des1, des2)

    # 确保我们有足够的匹配点进行计算
    assert len(matches) >= 2

    # 计算旋转角度
    angles = []
    for _ in range(1000):  # 重复20次
        # 随机选择两个匹配点
        m1, m2 = random.sample(matches, 2)

        # 计算每个匹配点在两个图像上的位置
        pt1_0 = kp1[m1.queryIdx].pt
        pt1_1 = kp2[m1.trainIdx].pt

        pt2_0 = kp1[m2.queryIdx].pt
        pt2_1 = kp2[m2.trainIdx].pt

        # 计算两个匹配点在两个图像上的角度
        angle_0 = np.arctan2(pt2_0[1] - pt1_0[1], pt2_0[0] - pt1_0[0])
        angle_1 = np.arctan2(pt2_1[1] - pt1_1[1], pt2_1[0] - pt1_1[0])

        # 计算角度差，并转换为度数
        angle_diff = np.degrees(angle_1 - angle_0)

        # 保证角度差在0到360度之间
        angle_diff = angle_diff % 360

        angles.append(angle_diff)

    # 去掉最大的五个和最小的五个角度
    angles = sorted(angles)[100:-100]

    # 计算平均旋转角度
    avg_angle = np.median(angles)
    angle = avg_angle
    if angle > 180:
        angle -= 360
    elif angle < -180:
        angle += 360

    return angle





def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 0.5)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

def process_image(image_path, min_area, max_aspect_ratio, threshold_value):
    image = image_path
    height, width, _ = image.shape
    roi = image[:, :int(width * 0.9)]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers_with_labels = []
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        area = cv2.contourArea(box)
        w, h = rect[1]

        if h != 0 and w != 0:
            aspect_ratio = max(w, h) / min(w, h)
        else:
            aspect_ratio = float('inf')

        if area > min_area and aspect_ratio < max_aspect_ratio:
            cv2.drawContours(roi, [box], 0, (0, 255, 0), 2)
            center = rect[0]

            x, y, w, h = cv2.boundingRect(contour)
            cropped_image = roi[int(y):int(y + h), int(x):int(x + w)]
            label = classify_object(cropped_image, model)

            # 计算旋转角度
            template = cv2.imread(f'baizhengcar/{label}_reference.jpg', 0)
            rotation_angle = calculate_rotation_angle(template, cropped_image)

            centers_with_labels.append(((center[0], center[1]), label, rotation_angle))
            cv2.putText(roi, label, (int(x), int(y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Result", roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return centers_with_labels




def move_arm_to_position(s, x, y, z, angle, D0, D2):
    try:
        position_command = pos(x, y, z, angle, D0, D2)
        s.write(position_command.encode())
    except ValueError as e:
        print(e)

def pos(x, y, z, angle, D0, D2):
    if (0 <= x <= 600) and (-600 <= y <= 0) and (x ** 2 + y ** 2 <= 360000) and (-92.5 <= z <= 0) and (
            -300 <= angle <= 300) and (D0 in [0, 1]) and (D2 in [0, 1]):
        position = f"[X:{x:.2f};Y:{y:.2f};Z:{z:.2f};A:{angle:.2f};D0:{D0};D2:{D2}]"
        return position
    else:
        raise ValueError("Error: Please check the coordinate range.")

def image_to_real_coordinate(image_x, image_y, scale_factor, image_origin_x, image_origin_y):
    real_x = -image_y * scale_factor + image_origin_x
    real_y = -image_x * scale_factor - image_origin_y
    return real_x, real_y

# def image_to_real_coordinate(image_x, image_y, scale_factor, image_origin_x, image_origin_y):
#     real_x = image_x * scale_factor + image_origin_x
#     real_y = -image_y * scale_factor - image_origin_y
#     return real_x, real_y


def calculate_position(i, first_x, first_y, offset_x, offset_y):
    x = first_x + i * offset_x
    y = first_y + i * offset_y
    return x, y


def goal(first_x, first_y, label):
    # print(f"Received label: {label}")
    k = label
    if k == 1:
        target_x = first_x
        target_y = first_y
    elif k == 2:
        target_x = first_x
        target_y = first_y - 45-1
    elif k == 3:
        target_x = first_x+1
        target_y = first_y - 90-2
    elif k == 4:
        target_x = first_x
        target_y = first_y - 129-7
    elif k == 5:
        target_x = first_x - 44-2
        target_y = first_y
    elif k == 6:
        target_x = first_x - 44-2
        target_y = first_y - 45-1
    elif k == 7:
        target_x = first_x - 44-2
        target_y = first_y - 90-2
    elif k == 8:
        target_x = first_x - 44-2
        target_y = first_y - 129-7
    elif k == 9:
        target_x = first_x - 88-4
        target_y = first_y
    elif k == 10:
        target_x = first_x - 88-4
        target_y = first_y - 45-1
    elif k == 11:
        target_x = first_x - 88-4
        target_y = first_y - 90-2
    elif k == 12:
        target_x = first_x - 88-4
        target_y = first_y - 129-7
    elif k ==13:
        target_x = first_x - 132-6
        target_y = first_y
    elif k == 14:
        target_x = first_x - 132-6
        target_y = first_y - 45-1
    elif k == 15:
        target_x = first_x - 132-6
        target_y = first_y - 90-2
    elif k == 16:
        target_x = first_x - 132-6
        target_y = first_y - 129-7
    else:
        raise ValueError("Invalid value for k.")

    return target_x, target_y


if __name__ == "__main__":
    min_area = 11000
    max_aspect_ratio = 2.0
    threshold_value = 40
    n_objects = 16  # 期望识别到的物体数量
    first_x, first_y = 440, -20  # 第一个物体移动到的位置
    # offset_x, offset_y = 0, -10  # 物体之间的偏移量
    camera_matrix = np.load("camera_matrix.npy")
    dist_coeffs = np.load("dist_coeffs.npy")
    label_to_number = {
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
        "10": 10,
        "11": 11,
        "12": 12,
        "13": 13,
        "14": 14,
        "15": 15,
        "16": 16,

    }

    # 图像坐标系到实际坐标系的转换参数
    scale_factor = 100/275  # 比例因子，根据实际情况调整
    image_origin_x = 388  # 图像坐标系的原点在实际坐标系中的x坐标
    image_origin_y = 148  # 图像坐标系的原点在实际坐标系中的y坐标
    # real_x = -image_y * scale_factor + image_origin_x
    # real_y = -image_x * scale_factor - image_origin_y

    # 打开串口
    s = serial.Serial("COM3", 115200, timeout=10)

    classified_objects = {}
    while len(classified_objects) < n_objects:
        # 打开摄像头
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # 捕获摄像头图像
        ret, frame = cap.read()
        undistorted_image = cv2.undistort(frame, camera_matrix, dist_coeffs)

        # 处理图像并获取物体中心点
        centers_with_labels = process_image(undistorted_image, min_area, max_aspect_ratio, threshold_value)

        # 关闭摄像头
        cap.release()

        for center, label, rotation_angle in centers_with_labels:
            real_x, real_y = image_to_real_coordinate(center[0], center[1], scale_factor, image_origin_x,
                                                      image_origin_y)
            if label not in classified_objects:
                classified_objects[label] = (real_x, real_y, rotation_angle)

        print(f"Detected {len(classified_objects)} unique objects, retrying if necessary...")

    # 控制机械臂移动
    # for i, (label, (real_x, real_y,rotation_angle)) in enumerate(classified_objects.items()):
    # for label, (real_x, real_y, rotation_angle) in classified_objects.items():
    for label_str, (real_x, real_y, rotation_angle) in classified_objects.items():
        label = int(label_str)

        # object_number = label_to_number[label]
        target_x, target_y = goal(first_x, first_y, label)

        # real_x, real_y = center
        # print(f"Object {label}: center at ({real_x}, {real_y}) in real coordinates")
        print(f"Moving {label} to position ({target_x}, {target_y})")
        print(center)
        # x, y = calculate_position(label, first_x, first_y, offset_x, offset_y)

        z = 0
        angle = rotation_angle
        D0 = 0
        D2 = 0

        # print(f"Moving {label} to position ({x}, {y}, {z})")
        print(f"Moving {label} to position ({target_x}, {target_y})")
        x = target_x
        y = target_y

        position_command = pos(real_x, real_y, z, 0, D0, D2)
        s.write(position_command.encode())
        time.sleep(2)

        # 夹具不吸
        D0 = 0
        D2 = 0
        position_command = pos(real_x, real_y, z, 0, D0, D2)
        s.write(position_command.encode())
        time.sleep(2)

        # 降低机械臂高度，以便夹具夹住物体
        z = -92.5
        position_command = pos(real_x, real_y, z, 0, D0, D2)
        s.write(position_command.encode())
        time.sleep(2)

        # 夹具吸
        D0 = 1
        D2 = 1
        position_command = pos(real_x, real_y, z, 0, D0, D2)
        s.write(position_command.encode())
        time.sleep(2)

        # 提高机械臂高度，将物体抬起
        z = 0

        position_command = pos(real_x, real_y, z, angle, D0, D2)
        s.write(position_command.encode())
        time.sleep(2)

        # 移动到目标位置
        position_command = pos(x, y, z, angle, D0, D2)
        s.write(position_command.encode())
        time.sleep(2)

        # 降低机械臂高度，将物体放到目标位置
        z = -92
        position_command = pos(x, y, z, angle, D0, D2)
        s.write(position_command.encode())
        time.sleep(2)

        # 夹具不吸，释放物体
        D0 = 0
        D2 = 0

        position_command = pos(x, y, z, angle, D0, D2)
        s.write(position_command.encode())
        time.sleep(2)

        # 夹具不吸，释放物体,回角度
        D0 = 0
        D2 = 0
        z = -60

        position_command = pos(x, y, z, angle, D0, D2)
        s.write(position_command.encode())
        time.sleep(2)

        # 提高机械臂高度，回到初始高度
        z = 0
        position_command = pos(x, y, z, 0, D0, D2)
        s.write(position_command.encode())
        time.sleep(2)
    position_command = pos(600, 0, 0, 0, 0, 0)
    s.write(position_command.encode())


    # 关闭串口
    s.close()

    print("Task completed.")
