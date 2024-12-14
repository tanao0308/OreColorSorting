import cv2
import os

# 从图像中检测轮廓并绘制矩形框，最后保存结果图像。
def detect_and_draw_rectangles(image_path, output_path, area_threshold=100):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法加载图像: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > area_threshold:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            result.append([x, y, w, h])
    cv2.imwrite(output_path, image)
    return result

def crop_and_save_regions(image_path, result, output_subfolder='output_rectangles', info_filename='regions_info.txt'):
    if not os.path.exists(output_subfolder):
        os.makedirs(output_subfolder)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法加载图像: {image_path}")
    info_file_path = os.path.join(output_subfolder, info_filename)
    with open(info_file_path, 'w') as f:
        f.write("Region Index, x, y, w, h\n")  # 写入表头
        for idx, (x, y, w, h) in enumerate(result):
            f.write(f"{idx+1}, {x}, {y}, {w}, {h}\n")
    print(f"位置信息已保存到 {info_file_path}")
    for idx, (x, y, w, h) in enumerate(result):
        cropped_image = image[y:y+h, x:x+w]
        cropped_image_path = os.path.join(output_subfolder, f"cropped_{idx+1}.png")
        cv2.imwrite(cropped_image_path, cropped_image)
    print(f"所有区域已保存至 {output_subfolder}")

if __name__ == "__main__":
    image_path = "../gendata/output_image.jpg"
    output_path = "particles-divide.png"
    result = detect_and_draw_rectangles(image_path, output_path)
    crop_and_save_regions(image_path, result)