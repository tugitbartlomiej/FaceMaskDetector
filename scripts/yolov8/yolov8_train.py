from ultralytics import YOLO
model = YOLO('yolov8n.pt')
# First model training
# model.train(data='/home/bartlomiej/Studia/Sem4/Przetwarzanie_Obrazów/Datasets/Mask_Classify.v7i.yolov8/data.yaml', epochs=100, imgsz=832, device=0)

# Second model training
model.train(data='/home/bartlomiej/Studia/Sem4/Przetwarzanie_Obrazów/face-masks/scripts/yolov8/dataset/data.yaml', epochs=150, imgsz=832, device=0)