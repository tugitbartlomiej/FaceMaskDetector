import os

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


path=os.getcwd()
path=os.path.join(path, 'dataset')

path_train = os.path.join(path, 'train')

path_test = os.path.join(path, 'test')

original_images_path = path_train

save_to_dir_path = '/home/bartlomiej/Studia/Sem4/Przetwarzanie Obrazów/Datasets/Augmented/'

# Przygotowanie przepływu danych i wskazanie, gdzie zapisać przerobione obrazy
train_generator = datagen.flow_from_directory(
    original_images_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    save_to_dir=save_to_dir_path,
    save_prefix='aug',  # Prefix dodawany do nazwy plików
    save_format='jpeg'  # Format zapisu obrazów
)

# Uruchomienie generowania i zapisywania obrazów
# Możesz zdecydować, ile obrazów chcesz wygenerować
# Na przykład 1000 obrazów
for i in range(1000):
    train_generator.next()
