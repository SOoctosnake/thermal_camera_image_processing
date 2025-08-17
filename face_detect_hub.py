# Author: Soham Kulkarni
# Date: 2025-05-15
# this script is to demonstrate that yolo11 face detection 
# does not work with thermal images


# load libraries
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from PIL import Image
import sys

# download model
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")

# load model
model = YOLO(model_path)

# inference
#image_path = "../thermal_dataset/soham_thermal/soham_thermal2.JPG"
image_path = sys.argv[1]  # Use command line argument for image path
model.save("yolo8_facedetect.pt")
output = model(Image.open(image_path))
results = Detections.from_ultralytics(output[0])
mresults = model.predict(source=image_path, save=True, save_txt=True, save_conf=True)
print(results)