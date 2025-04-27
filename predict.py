import torch
from torchvision import transforms
from PIL import Image
from config import resize_x
from model import CustomResNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inference transform (same as validation)
infer_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(resize_x),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(list_of_img_paths):
    model = CustomResNet()
    model.load_state_dict(torch.load("checkpoints/final_weights.pth", map_location=device))
    model = model.to(device)
    model.eval()

    images = []
    for path in list_of_img_paths:
        img = Image.open(path).convert("RGB")
        img = infer_transform(img)
        images.append(img)

    batch = torch.stack(images).to(device)
    with torch.no_grad():
        output = model(batch)
        preds = output.argmax(dim=1).cpu().tolist()

    return preds
