import cv2
import numpy as np

# 读取图像
image = cv2.imread(r'C:\Users\admin\Desktop\syn\colors\pink1.jpg')

# 将图像转换为灰度
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 进行边缘检测
edges = cv2.Canny(gray, threshold1=180, threshold2=600)

# 查找轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 遍历轮廓，得到每个矩形的边界框
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)

    # 画出矩形框
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 保存带有矩形边界框的图像
cv2.imwrite(r'C:\Users\admin\Desktop\syn\result\pink1-.jpg', image)

# 显示图像
cv2.imshow('Contours with Rectangles', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
