import cv2

# 从图像中检测轮廓并绘制矩形框，最后保存结果图像。
def detect_and_draw_rectangles(image_path, output_path, area_threshold=100):
    # 1. 加载图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法加载图像: {image_path}")
    # 2. 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 3. 预处理（高斯模糊和边缘检测）
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    # 4. 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 5. 绘制矩形框
    result = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > area_threshold:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            result.append([x, y, w, h])
    cv2.imwrite(output_path, image)
    return result
    # cv2.imshow("Rectangles", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "../gendata/output_image.jpg"
    output_path = "particles-divide.png"
    detect_and_draw_rectangles(image_path, output_path)
