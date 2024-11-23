import cv2
import numpy as np
import random

# 参数设置
image_width = 500  # 图像宽度
image_height = 500  # 图像高度
min_radius = 10  # 圆的最小半径
max_radius = 30  # 圆的最大半径
target_density = 0.3  # 目标密度（30%）

# 创建空白图像
image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
total_area = image_width * image_height
circle_area = 0  # 当前已填充的圆的总面积

# 存储圆的信息
circles = []

# 随机生成圆
while circle_area / total_area < target_density:
    # 随机生成圆心和半径
    radius = random.randint(min_radius, max_radius)
    center_x = random.randint(radius, image_width - radius)
    center_y = random.randint(radius, image_height - radius)

    # 检查是否与现有圆相交
    overlap = False
    for cx, cy, r in circles:
        distance = ((center_x - cx) ** 2 + (center_y - cy) ** 2) ** 0.5
        if distance < radius + r:  # 两圆距离小于半径和，说明相交
            overlap = True
            break

    if not overlap:
        # 绘制实心圆
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))  # 随机颜色
        cv2.circle(image, (center_x, center_y), radius, color, -1)

        # 更新圆的信息
        circles.append((center_x, center_y, radius))
        circle_area += np.pi * radius ** 2  # 更新总面积

# 显示图像
# cv2.imshow("Random Circles", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 如果需要保存图像
cv2.imwrite("particles.png", image)
