import os
import cv2
from roboflow import Roboflow
import supervision as sv

# Set up your Roboflow model
api_key = "JxCMjosaIa8gVEhzGMq6"
rf = Roboflow(api_key=api_key)
project = rf.workspace().project("mask-aify8")
model = project.version(3).imagenet_model

image_dir = "/home/bartlomiej/Studia/Sem4/Przetwarzanie_Obrazów/face-masks/scripts/dataset/train/with_mask/"

for filename in os.listdir(image_dir):
    if filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(image_dir, filename)

        result = model.predict(image_path, confidence=40, overlap=30).json()

        labels = [item["class"] for item in result["predictions"]]
        detections = sv.Detections.from_roboflow(result)

        label_annotator = sv.LabelAnnotator()
        bounding_box_annotator = sv.BoxAnnotator()

        image = cv2.imread(image_path)

        annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

        sv.plot_image(image=annotated_image, size=(16, 16))
    else:
        continue
