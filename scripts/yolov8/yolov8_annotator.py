import cv2
import numpy as np
from ultralytics import YOLO
import os


def write_annotation(file_path, box, image_width, image_height):
    x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
    class_id = box.cls[0].item()
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    x_center_n = round(x_center / image_width, 8)
    y_center_n = round(y_center / image_height, 8)
    width_n = round(width / image_width, 8)
    height_n = round(height / image_height, 8)

    with open(file_path, 'w') as f:
        f.write(f"{class_id} {x_center_n} {y_center_n} {width_n} {height_n}\n")


def is_image_in_dataset(image_name, dataset_dir):
    """Sprawdza, czy obraz jest już w zestawie treningowym"""
    for split in ['train', 'valid', 'test']:
        if os.path.exists(os.path.join(dataset_dir, split, 'images', image_name)):
            return True
    return False


# Load your YOLO model with custom or pre-trained weights
model = YOLO('runs/detect/train2/weights/best.pt')

input_image_dir = '/home/bartlomiej/Studia/Sem4/Przetwarzanie_Obrazów/Datasets/with_mask'
base_dataset_dir = './dataset'

# Create subdirectories for train, valid, test, their images, and labels
for split in ['train', 'valid', 'test', 'not_detected']:
    os.makedirs(os.path.join(base_dataset_dir, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(base_dataset_dir, split, 'labels'), exist_ok=True)

os.makedirs(os.path.join(base_dataset_dir, 'not_detected', 'not_detected_images'), exist_ok=True)

images = os.listdir(input_image_dir)

for image_name in images:
    if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    if is_image_in_dataset(image_name, base_dataset_dir):
        print(f'Image {image_name} is already in the dataset.')
        continue

    image_path = os.path.join(input_image_dir, image_name)
    image = cv2.imread(image_path)
    image_height, image_width = image.shape[:2]

    results = model.predict(image)

    # Show only the first detected box, if any
    if len(results[0].boxes) > 0:

        # Iterate through all detected boxes
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            label = results[0].names[box.cls[0].item()]
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow('image', image)
        key = cv2.waitKey(0)

        image = cv2.imread(image_path)

        if key == 13:  # Enter key
            # Save to training set
            annotated_image_path = os.path.join(base_dataset_dir, 'train', 'images', image_name)
            annotation_file_path = os.path.join(base_dataset_dir, 'train', 'labels',
                                                os.path.splitext(image_name)[0] + '.txt')
            write_annotation(annotation_file_path, box, image_width, image_height)
            cv2.imwrite(annotated_image_path, image)

        elif key == 32:  # Space key
            # Save to not_detected
            not_detected_image_path = os.path.join(base_dataset_dir, 'not_detected', 'not_detected_images', image_name)
            cv2.imwrite(not_detected_image_path, image)

        cv2.destroyAllWindows()
    else:
        # If no box is detected, save to not_detected
        not_detected_image_path = os.path.join(base_dataset_dir, 'not_detected', 'not_detected_images', image_name)
        cv2.imwrite(not_detected_image_path, image)

print('Process completed.')
