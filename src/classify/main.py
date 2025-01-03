import torch
from PIL import Image
import torchvision.transforms as transforms

from classify.model import CNN

def classify_image(model, image_path, transform=None):
    if transform is None:
        transform = transforms.Compose([
                transforms.Resize((250,250)) ,
                transforms.ToTensor() ,
                transforms.Normalize((0),(1))])
    img = Image.open(image_path)
    img = transform(img).unsqueeze(0)
    model.eval()
    # 禁用梯度计算
    with torch.no_grad():
        output = model(img)
        predicted_class = output.argmax(dim=1)
    return predicted_class.item()

if __name__ == "__main__":
    model = CNN(5)
    model.load_state_dict(torch.load('trained_model.pth'))

    image_path = 'Arborio (12).jpg'
    predicted_class = classify_image(model, image_path)
    print(f'Predicted Class: {predicted_class}')
