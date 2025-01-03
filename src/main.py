import os
import torch

from split.main import detect_and_draw_rectangles, crop_and_save_regions
from classify.main import classify_image, CNN


# split
image_path = "./gendata/output_image.jpg"
output_path = "particles-divide.png"
result = detect_and_draw_rectangles(image_path, output_path)
crop_and_save_regions(image_path, result)

# classify
model = CNN(5)
model.load_state_dict(torch.load('trained_model.pth'))
directory = 'output_rectangles'
for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        predicted_class = classify_image(model, file_path)
        print(f'File: {filename}, Predicted Class: {predicted_class}')









