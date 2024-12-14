import cv2
from PIL import Image
import numpy as np
import os
import random

# 从 opencv 格式图像中裁剪出目标物体
def extract_and_crop_object(opencv_image: np.ndarray, output_path="output_image_cropped.jpg"):
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_img = opencv_image[y:y+h, x:x+w]
    return cropped_img

def pil_to_opencv(pil_image: Image) -> np.ndarray:
    open_cv_image = np.array(pil_image)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    return open_cv_image

def opencv_to_pil(opencv_image: np.ndarray) -> Image:
    rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    return pil_image

# 从指定文件夹中随机选择一张图片并返回其路径。
def select_random_image(folder_path):
    # 获取所有图片文件的路径（假设图片文件扩展名为 .jpg, .jpeg, .png, .bmp, .gif 等）
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'gif'))]
    if not image_files:
        raise ValueError("该文件夹中没有图片文件！")
    random_image = random.choice(image_files)
    return os.path.join(folder_path, random_image)