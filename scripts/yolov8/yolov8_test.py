import cv2
from ultralytics import YOLO
import os

model = YOLO('runs/detect/train3/weights/best.pt')

source_dir = '/home/bartlomiej/Studia/Sem4/Przetwarzanie_Obrazów/face-masks/scripts/dataset/test/with_mask'

# # Directory where the images with detections will be saved
# output_dir = 'with_mask_detected/'
# os.makedirs(output_dir, exist_ok=True)
#
# # Process all images in the directory
# for image_name in os.listdir(source_dir):
#     if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
#         image_path = os.path.join(source_dir, image_name)
#         image = cv2.imread(image_path)
#         results = model.predict(image)
#
#         # Draw bounding boxes and labels on the image
#         for box in results[0].boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
#             label = results[0].names[box.cls[0].item()]
#             image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
#             image = cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
#
#         # Save the image with detections
#         cv2.imwrite(os.path.join(output_dir, image_name), image)
#
# print('All images have been processed and saved with detections.')


model.eval()
results = model(source_dir)

results.print()

