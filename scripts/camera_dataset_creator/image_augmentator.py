import os
import numpy as np
import cv2
from keras_preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator

input_folder = 'movie_frames'

main_output_folder = 'metamorphic_transforms'
os.makedirs(main_output_folder, exist_ok=True)

rotation_range = range(0, 45, 5)
zoom_range = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
illumination_changes = [20, 40, 60, 80, 100, 120]
movement_offsets = range(0, 21, 5)
blurring_kernels = [3, 5, 7, 9, 11, 13]
scaling_factors = [0.5, 0.75, 1.0, 1.25, 1.5]

# Create subfolders for each transformation parameter
for param in rotation_range:
    os.makedirs(os.path.join(main_output_folder, f'rotation_{param}_degree'), exist_ok=True)

for param in zoom_range:
    os.makedirs(os.path.join(main_output_folder, f'zoom_{param}'), exist_ok=True)

for param in illumination_changes:
    os.makedirs(os.path.join(main_output_folder, f'illumination_{param}'), exist_ok=True)

for param in movement_offsets:
    os.makedirs(os.path.join(main_output_folder, f'movement_{param}_pixels'), exist_ok=True)

for param in blurring_kernels:
    os.makedirs(os.path.join(main_output_folder, f'blurring_{param}_kernel'), exist_ok=True)

for param in scaling_factors:
    os.makedirs(os.path.join(main_output_folder, f'scaling_{param}'), exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith('.png'):
        img_path = os.path.join(input_folder, filename)
        img = load_img(img_path)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)

        # Load original image using OpenCV for other transformations
        original_img = cv2.imread(img_path)

        # Apply rotation
        for i in rotation_range:
            datagen = ImageDataGenerator(rotation_range=i)
            iterator = datagen.flow(x, batch_size=1, save_to_dir=os.path.join(main_output_folder, f'rotation_{i}_degree'),
                                    save_prefix=f'rotation_{i}_degree_{filename}', save_format='png')
            iterator.next()

        # Apply zoom
        for zoom in zoom_range:
            datagen = ImageDataGenerator(zoom_range=[zoom, zoom])
            iterator = datagen.flow(x, batch_size=1, save_to_dir=os.path.join(main_output_folder, f'zoom_{zoom}'),
                                    save_prefix=f'zoom_{zoom}_{filename}', save_format='png')
            iterator.next()

        # Apply illumination change
        for change in illumination_changes:
            brightness_matrix = np.ones(original_img.shape, dtype='uint8') * change
            illuminated_img = cv2.add(original_img, brightness_matrix)
            cv2.imwrite(os.path.join(main_output_folder, f'illumination_{change}', f'illumination_{change}_{filename}'), illuminated_img)

        # Apply movement
        for offset in movement_offsets:
            M = np.float32([[1, 0, offset], [0, 1, offset]])  # Move by offset pixels
            moved_img = cv2.warpAffine(original_img, M, (original_img.shape[1], original_img.shape[0]))
            cv2.imwrite(os.path.join(main_output_folder, f'movement_{offset}_pixels', f'movement_{offset}_pixels_{filename}'), moved_img)

        # Apply blurring
        for kernel_size in blurring_kernels:
            blurred_img = cv2.GaussianBlur(original_img, (kernel_size, kernel_size), 0)
            cv2.imwrite(os.path.join(main_output_folder, f'blurring_{kernel_size}_kernel', f'blurring_{kernel_size}_kernel_{filename}'), blurred_img)

        # Apply scaling
        for scale in scaling_factors:
            scaled_img = cv2.resize(original_img, None, fx=scale, fy=scale)
            cv2.imwrite(os.path.join(main_output_folder, f'scaling_{scale}', f'scaling_{scale}_{filename}'), scaled_img)


print("Finished processing images.")
