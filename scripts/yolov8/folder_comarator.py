import os

train_images_dir = './dataset/train/images'
train_labels_dir = './dataset/train/labels'

image_files = set(os.path.splitext(file)[0] for file in os.listdir(train_images_dir) if file.endswith(('.png', '.jpg', '.jpeg')))

label_files = set(os.path.splitext(file)[0] for file in os.listdir(train_labels_dir) if file.endswith('.txt'))

for image_file in image_files:
    if image_file not in label_files:
        os.remove(os.path.join(train_images_dir, image_file + '.jpg'))  # Usuń plik obrazu
        print(f"Removed image: {image_file}.jpg")
