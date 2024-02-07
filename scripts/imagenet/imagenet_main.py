import os

from PIL import UnidentifiedImageError

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
 # Używa tylko jednego, określonego GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print("GPU available")
    try:
        # Ustawia maksymalne użycie pamięci na 50% dostępnej pamięci GPU
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5732 // 2)]
        )
    except RuntimeError as e:
        # Błąd wywołany gdy konfiguracja pamięci jest ustawiona po inicjalizacji GPU
        print(e)

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf

import cv2
import numpy as np
import concurrent.futures
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input


class ImageNetImageDetector:
    def __init__(self, model_path, base_source_dir, base_output_dir):
        self.model = load_model(model_path)
        self.base_source_dir = base_source_dir
        self.base_output_dir = base_output_dir

        os.makedirs(self.base_output_dir, exist_ok=True)
        self.subdirectories = [d for d in os.listdir(self.base_source_dir) if
                               os.path.isdir(os.path.join(self.base_source_dir, d))]
        for subdir in self.subdirectories:
            os.makedirs(os.path.join(self.base_output_dir, subdir), exist_ok=True)


    @staticmethod
    def preprocess_image(img_path):
        try:
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            return img_array
        except UnidentifiedImageError:
            print(f"Nie można zidentyfikować obrazu: {img_path}. Obraz może być uszkodzony.")
            return None

    def process_images(self, subdir):
        source_dir = os.path.join(self.base_source_dir, subdir)
        output_dir = os.path.join(self.base_output_dir, subdir)

        results_list = []
        for image_name in os.listdir(source_dir):
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                print('Processing image: ', image_name)
                image_path = os.path.join(source_dir, image_name)
                output_image_path = os.path.join(output_dir, image_name)
                if os.path.exists(output_image_path):
                    continue

                processed_image = self.preprocess_image(image_path)
                prediction = self.model.predict(processed_image)
                detected = 'No'
                original_img = cv2.imread(image_path)  # Read the original image using OpenCV
                if prediction[0][0] > 0.5:
                    detected = 'Yes'
                    text = "Mask Score: {:.2f}".format(prediction[0][0])
                    cv2.putText(original_img, text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)
                    print(text)
                else:
                    text = "No Mask. Score: {:.2f}".format(prediction[0][0])
                    print(text)
                    cv2.putText(original_img, text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)

                cv2.imwrite(output_image_path, original_img)

                results_list.append({'Folder': subdir, 'Image': image_name, 'Detected': detected})

        return results_list

    # def run_detection(self):
    #     all_results = []
    #     with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
    #         futures = [executor.submit(self.process_images, subdir) for subdir in self.subdirectories]
    #         for future in concurrent.futures.as_completed(futures):
    #             all_results.extend(future.result())
    #     return all_results

    def run_detection(self):
        all_results = []
        for subdir in self.subdirectories:
            results = self.process_images(subdir)
            all_results.extend(results)
        self.plot_results(all_results)
        return all_results

    def plot_results(self, detection_results):
        detection_log_df = pd.DataFrame(detection_results)
        if not detection_log_df.empty:
            plt.figure(figsize=(15, 6))

            detection_summary = detection_log_df.groupby(['Folder', 'Detected']).size().unstack(fill_value=0)

            ax = detection_summary.plot(kind='bar', stacked=False)
            plt.title('Liczba wykrytych i niewykrytych masek w każdym folderze')
            plt.xlabel('Folder')
            plt.ylabel('Liczba obrazów')

            plt.xticks(rotation=90, ha='center')  # Ha to horizontal alignment
            plt.tight_layout()  # Dostosowanie layoutu, aby pomieścić etykiety

            plt.savefig(os.path.join(self.base_output_dir, 'detection_results_summary.png'))
            plt.close()
        else:
            print("No data available for plotting.")


# # # Usage example
# detector = ImageNetImageDetector()
# detection_results = detector.run_detection()
# detector.plot_results(detection_results)
