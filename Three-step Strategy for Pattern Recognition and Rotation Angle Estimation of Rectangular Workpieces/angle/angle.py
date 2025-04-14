import cv2
import numpy as np
import random

def calculate_rotation_angle(template_path, actual_image_path):
    # 加载图像
    template = cv2.imread(template_path, 0)
    actual_image = cv2.imread(actual_image_path, 0)

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
    for _ in range(1000):  # 重复1000次
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

    return angles, angle
if __name__ == "__main__":

    angles, avg_angle = calculate_rotation_angle(r'C:\Users\Administrator\Desktop\shiyan\1\1-600\0.jpg', r'C:\Users\Administrator\Desktop\shiyan\1\1-600\6.jpg')
    print("All calculated angles: ", angles)
    print("Average angle: ", avg_angle)
