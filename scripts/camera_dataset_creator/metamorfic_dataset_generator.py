import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import load_img, img_to_array


class MetamorphicDatasetCreator:
    def __init__(self, input_video_path, frames_output_folder, augmented_images_folder):
        self.input_video_path = input_video_path
        self.frames_output_folder = frames_output_folder
        self.augmented_images_folder = augmented_images_folder

        # Ensure output directories exist
        os.makedirs(self.frames_output_folder, exist_ok=True)
        os.makedirs(self.augmented_images_folder, exist_ok=True)

    def trim_frames(self):
        cap = cv2.VideoCapture(self.input_video_path)
        if not cap.isOpened():
            print("Error: Cannot open video.")
            return

        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_number % 10 == 0:  # Adjust this value to control frame sampling rate
                frame_path = os.path.join(self.frames_output_folder, f"frame{frame_number:04d}.png")
                cv2.imwrite(frame_path, frame)
            frame_number += 1

        cap.release()
        cv2.destroyAllWindows()
        print(f"Saved {frame_number} frames.")

    def augment_images(self, rotation_range, zoom_range, illumination_changes, movement_offsets, blurring_kernels,
                       scaling_factors):
        # Iterate over each frame in the frames output folder
        for filename in os.listdir(self.frames_output_folder):
            # Create subdirectories for each transformation type
            for i in rotation_range:
                os.makedirs(os.path.join(self.augmented_images_folder, f'rotation_{i}_degree'), exist_ok=True)
            for zoom in zoom_range:
                os.makedirs(os.path.join(self.augmented_images_folder, f'zoom_{zoom}'), exist_ok=True)
            for change in illumination_changes:
                os.makedirs(os.path.join(self.augmented_images_folder, f'illumination_{change}'), exist_ok=True)
            for offset in movement_offsets:
                os.makedirs(os.path.join(self.augmented_images_folder, f'movement_{offset}_pixels'), exist_ok=True)
            for kernel_size in blurring_kernels:
                os.makedirs(os.path.join(self.augmented_images_folder, f'blurring_{kernel_size}_kernel'), exist_ok=True)
            for scale in scaling_factors:
                os.makedirs(os.path.join(self.augmented_images_folder, f'scaling_{scale}'), exist_ok=True)

            if filename.endswith('.png'):
                img_path = os.path.join(self.frames_output_folder, filename)
                img = load_img(img_path)  # Loading the image
                x = img_to_array(img)  # Converting to a numpy array
                x = np.expand_dims(x, axis=0)  # Reshaping to (1, height, width, channels)

                # Load original image using OpenCV for other transformations
                original_img = cv2.imread(img_path)

                print("Creating rotated images")
                for i in rotation_range:
                    datagen = ImageDataGenerator(rotation_range=i)
                    iterator = datagen.flow(x, batch_size=1,
                                            save_to_dir=os.path.join(self.augmented_images_folder,
                                                                     f'rotation_{i}_degree'),
                                            save_prefix=f'rotation_{i}_degree_{filename}', save_format='png')
                    iterator.next()

                # Apply zoom
                print("Creating zoomed images")
                for zoom in zoom_range:
                    datagen = ImageDataGenerator(zoom_range=[zoom, zoom])
                    iterator = datagen.flow(x, batch_size=1,
                                            save_to_dir=os.path.join(self.augmented_images_folder, f'zoom_{zoom}'),
                                            save_prefix=f'zoom_{zoom}_{filename}', save_format='png')
                    iterator.next()

                print("Creating illuminated images")
                for change in illumination_changes:
                    brightness_matrix = np.ones(original_img.shape, dtype='uint8') * change
                    illuminated_img = cv2.add(original_img, brightness_matrix)
                    cv2.imwrite(
                        os.path.join(self.augmented_images_folder, f'illumination_{change}',
                                     f'illumination_{change}_{filename}'),
                        illuminated_img)

                print("Creating moved images")
                for offset in movement_offsets:
                    M = np.float32([[1, 0, offset], [0, 1, offset]])  # Move by offset pixels
                    moved_img = cv2.warpAffine(original_img, M, (original_img.shape[1], original_img.shape[0]))
                    cv2.imwrite(os.path.join(self.augmented_images_folder, f'movement_{offset}_pixels',
                                             f'movement_{offset}_pixels_{filename}'), moved_img)

                print("Creating blurred images")
                for kernel_size in blurring_kernels:
                    blurred_img = cv2.GaussianBlur(original_img, (kernel_size, kernel_size), 0)
                    cv2.imwrite(os.path.join(self.augmented_images_folder, f'blurring_{kernel_size}_kernel',
                                             f'blurring_{kernel_size}_kernel_{filename}'), blurred_img)

                print("Creating scaled images")
                for scale in scaling_factors:
                    scaled_img = cv2.resize(original_img, None, fx=scale, fy=scale)
                    cv2.imwrite(
                        os.path.join(self.augmented_images_folder, f'scaling_{scale}', f'scaling_{scale}_{filename}'),
                        scaled_img)

# creator = MetamorphicDatasetCreator('path/to/input/video.mp4', 'path/to/frames/output', 'path/to/augmented/images')
# creator.trim_frames()
# creator.augment_images(rotation_range=[0, 10, 20], zoom_range=[1.0, 1.5], illumination_changes=[20, 40], movement_offsets=[5, 10], blurring_kernels=[3, 5], scaling_factors=[0.75, 1.0])
