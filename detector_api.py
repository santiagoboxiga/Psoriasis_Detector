from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import uuid
import os
from ultralytics import YOLO
from detector import load_u2net, predict_with_filters_and_classes

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Update this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models once
yolo_model = YOLO("psoriasis_detector.pt").to("cuda")
u2net_model = load_u2net()

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    filename = f"temp_uploads/{uuid.uuid4().hex}_{file.filename}"
    with open(filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        detections = predict_with_filters_and_classes(filename, yolo_model, u2net_model)
    finally:
        os.remove(filename)  # Cleanup temp file

    # Check detections
    for detection in detections:
        if detection["class"] == 0 and float(detection["confidence"].strip('%')) >= 70:
            return {"result": "psoriasis"}

    return {"result": "healthy"}

