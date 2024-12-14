import random
from PIL import Image
from utils import *

# 加载所有待处理的图片
def load_images(image_paths):
    return [Image.open(img_path) for img_path in image_paths]

# 随机旋转图片
def random_rotate(img):
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
    image_paths = []
    for i in range(num_images):
        image_paths.append(select_random_image(random.choice(folder_paths)))
    images = load_images(image_paths)
    new_size = (50, 50)
    images = [img.resize(new_size) for img in images]
    canvas = Image.new('RGB', canvas_size, (0, 0, 0))
    placed_images = []
    for _ in range(num_images):
        img = random.choice(images)
        img = opencv_to_pil(extract_and_crop_object(pil_to_opencv(img)))
        img = random_rotate(img)
        img_width, img_height = img.size
        max_x = canvas_size[0] - img_width
        max_y = canvas_size[1] - img_height
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        while check_overlap(img, placed_images, (x, y)):
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
        canvas.paste(img, (x, y))
        placed_images.append((img, (x, y)))
    canvas.save(output_path)
    canvas.show()

if __name__ == "__main__":
    folder_paths = ["../data/RiceImagesDataset/Arborio", "../data/RiceImagesDataset/Basmati"]
    output_path = "output_image.jpg"
    place_images_on_black_background(folder_paths, output_path, canvas_size=(500, 500), num_images=50)
