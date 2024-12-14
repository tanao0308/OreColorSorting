import random
from PIL import Image

from utils import *

# 加载所有待处理的图片
def load_images(image_paths):
    return [Image.open(img_path) for img_path in image_paths]
# 随机旋转图片
def random_rotate(img):
    # 随机选择一个角度进行旋转
    angle = random.randint(-90, 90)
    return img.rotate(angle, expand=True)  # 使用 expand=True 保证旋转后不会裁剪图片

# 检查图片是否与已放置的图片重叠
def check_overlap(new_img, placed_images, position):
    new_img_width, new_img_height = new_img.size
    new_x, new_y = position
    for img, (x, y) in placed_images:
        img_width, img_height = img.size
        if (new_x < x + img_width and new_x + new_img_width > x and
            new_y < y + img_height and new_y + new_img_height > y):
            return True
    return False

# 随机选择图像并粘贴到背景上
def place_images_on_black_background(folder_paths, output_path, canvas_size=(2000, 2000), num_images=5):
    # 加载所有图片
    image_paths = []
    for i in range(num_images):
        image_paths.append(select_random_image(random.choice(folder_paths)))
    images = load_images(image_paths)
    # 设置缩放大小
    new_size = (50, 50)
    images = [img.resize(new_size) for img in images]
    # 创建一个黑色背景
    canvas = Image.new('RGB', canvas_size, (0, 0, 0))
    # 记录已放置的图片和它们的坐标
    placed_images = []
    for _ in range(num_images):
        # 随机选择一张图片
        img = random.choice(images)
        # 截取图像中的 rice 主体
        img = opencv_to_pil(extract_and_crop_object(pil_to_opencv(img)))
        # 随机旋转图片
        img = random_rotate(img)
        # 随机生成图片的位置
        img_width, img_height = img.size
        max_x = canvas_size[0] - img_width
        max_y = canvas_size[1] - img_height
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        # 检查是否与已放置的图片重叠
        while check_overlap(img, placed_images, (x, y)):
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
        # 将图片粘贴到背景上
        canvas.paste(img, (x, y))
        # 记录图片位置
        placed_images.append((img, (x, y)))
    # 保存最终的图片
    canvas.save(output_path)
    canvas.show()

if __name__ == "__main__":
    folder_paths = ["../data/RiceImagesDataset/Arborio", "../data/RiceImagesDataset/Basmati"]
    output_path = "output_image.jpg"
    place_images_on_black_background(folder_paths, output_path, canvas_size=(500, 500), num_images=50)
