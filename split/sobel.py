import cv2

# 1. 加载图像
image = cv2.imread("particles.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2. 预处理
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)

# 3. 提取轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 4. 绘制矩形框
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 100:  # 面积阈值
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 5. 显示结果
cv2.imshow("Rectangles", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 如果需要保存图像
cv2.imwrite("particles-divide.png", image)