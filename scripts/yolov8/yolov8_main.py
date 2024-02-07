import cv2
from ultralytics import YOLO
import os
import concurrent.futures
import pandas as pd
import matplotlib.pyplot as plt

class Yolov8ImageDetector:
    def __init__(self, model_path, base_source_dir, base_output_dir):
        self.model = YOLO(model_path)
        self.base_source_dir = base_source_dir
        self.base_output_dir = base_output_dir

        os.makedirs(self.base_output_dir, exist_ok=True)
        self.subdirectories = [d for d in os.listdir(self.base_source_dir) if
                               os.path.isdir(os.path.join(self.base_source_dir, d))]
        for subdir in self.subdirectories:
            os.makedirs(os.path.join(self.base_output_dir, subdir), exist_ok=True)

    def process_images(self, subdir):
        source_dir = os.path.join(self.base_source_dir, subdir)
        output_dir = os.path.join(self.base_output_dir, subdir)
        os.makedirs(output_dir, exist_ok=True)

        results_list = []
        for image_name in os.listdir(source_dir):
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(source_dir, image_name)
                output_image_path = os.path.join(output_dir, image_name)
                if os.path.exists(output_image_path):
                    continue
                image = cv2.imread(image_path)
                results = self.model.predict(image)
                detected = 'No'
                if results[0].boxes:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        label = results[0].names[box.cls[0].item()]
                        if label == 'Mask':
                            image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            image = cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                            detected = 'Yes'
                cv2.imwrite(output_image_path, image)
                results_list.append({'Folder': subdir, 'Image': image_name, 'Detected': detected})
        return results_list

    def run_detection(self):
        all_results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(self.process_images, subdir) for subdir in self.subdirectories]
            for future in concurrent.futures.as_completed(futures):
                all_results.extend(future.result())
        self.plot_results(all_results)
        return all_results

    def plot_results(self, detection_results):
        detection_log_df = pd.DataFrame(detection_results)
        if not detection_log_df.empty:
            for folder in detection_log_df['Folder'].unique():
                folder_df = detection_log_df[detection_log_df['Folder'] == folder]
                detection_counts = folder_df['Detected'].value_counts()
                plt.figure(figsize=(10, 6))
                detection_counts.plot(kind='bar')
                plt.title(f'Detection Results for {folder}')
                plt.xlabel('Detected Masks')
                plt.ylabel('Number of Images')
                plt.savefig(os.path.join(self.base_output_dir, f'detection_results_{folder}.png'))
                plt.close()
        else:
            print("No data available for plotting.")

# # Usage
# detector = Yolov8ImageDetector()
# detection_results = detector.run_detection()
# detector.plot_results(detection_results)
