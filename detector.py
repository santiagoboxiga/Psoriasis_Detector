import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from U2NET.model import U2NET  # from U-2-Net/model.py

def load_u2net():
    model = U2NET(3, 1)
    model.load_state_dict(torch.load("U2NET/saved_models/u2net/u2net.pth", map_location='cpu'))
    model.eval()
    return model

def get_u2net_mask(image_path, model):
    image = Image.open(image_path).convert('RGB')
    original_size = image.size

    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)[0]
        mask = output.squeeze().cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        mask = cv2.resize(mask, original_size)
        _, binary = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
    return binary

def crop_with_mask(image_path, mask):
    image = cv2.imread(image_path)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return image[y:y+h, x:x+w]

def predict_with_filters(image_path, yolo_model, u2net_model, conf_thresh=0.7, min_area=500):
    mask = get_u2net_mask(image_path, u2net_model)
    cropped = crop_with_mask(image_path, mask)

    results = yolo_model.predict(cropped)[0]

    confidences = []
    for box in results.boxes:
        conf = round(box.conf.item() * 100, 2)
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        area = (x2 - x1) * (y2 - y1)
        if conf >= conf_thresh * 100 and area >= min_area:
            confidences.append(f"{conf}%")

    # Save the annotated result (optional, can disable)
    #annotated = results.plot()
    #cv2.imwrite("filtered_result.jpg", annotated)

    return confidences
